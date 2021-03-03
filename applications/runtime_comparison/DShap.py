import matplotlib
matplotlib.use('Agg')
from applications.runtime_comparison.shap_utils import *
import warnings
import pickle as pkl
import scipy
import copy
import torch
import os


def delete_rows_csr(mat, index):
    if not isinstance(mat, scipy.sparse.csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    mask = np.ones(mat.shape[0], dtype=bool)
    mask[index] = False
    return mat[mask]



class DShap(object):

    def __init__(self, X_train, y_train, X_test, y_test, model_family, metric, X_val=None, y_val=None, X_train_deep=None, X_test_deep=None,
                 sources=None, directory="../data_valuation/", problem='classification', seed=None, nodump=False, **kwargs):
        """
        Args:
            X_train: Train covariates
            y_train: Train labels
            X_test: Test covariates
            y_test: Test labels
            model_family: The model family used for learning algorithm
            metric: Evaluation metric
            X_val: Validation covariates
            y_val: Validation labels
            X_train_deep: Train deep features
            X_test_deep: Test deep features
            sources: An array or dictionary assigning each point to its group.
                If None, evey points gets its individual value.
            directory: Directory to save results and figures.
            problem: "Classification"
            seed: Random seed. When running parallel monte-carlo samples,
                we initialize each with a different seed to prevent getting
                same permutations.
            **kwargs: Arguments of the model
        """

        if seed is not None:
            np.random.seed(seed)
        self.problem = problem
        self.model_family = model_family
        self.metric = metric
        self.directory = directory
        self.hidden_units = kwargs.get('hidden_layer_sizes', [])
        self.nodump = nodump
        if self.directory is not None:
            if not os.path.exists(directory):
                os.makedirs(directory)
                os.makedirs(os.path.join(directory, 'weights'))
                os.makedirs(os.path.join(directory, 'plots'))
            self._initialize_instance(X_train, y_train, X_val, y_val, X_test, y_test, X_train_deep, X_test_deep, sources)

        self.model = return_model(self.model_family, **kwargs)
        self.random_score = self.init_score(self.metric)
        self.is_regression = False
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def _initialize_instance(self, X_train, y_train, X_val, y_val, X_test, y_test, X_train_deep, X_test_deep, sources=None):
        """loads or creates data"""

        if sources is None:
            sources = {i: np.array([i]) for i in range(X_train.shape[0])}
        elif not isinstance(sources, dict):
            sources = {i: np.where(sources == i)[0] for i in set(sources)}
        self.sources = sources
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.X_test, self.y_test = X_test, y_test
        self.X_train_deep = X_train_deep
        self.X_test_deep = X_test_deep
        self.vals_loo = None
        previous_results = os.listdir(self.directory)
        tmc_numbers = [int(name.split('.')[-2].split('_')[-1])
                       for name in previous_results if 'mem_tmc' in name]
        g_numbers = [int(name.split('.')[-2].split('_')[-1])
                     for name in previous_results if 'mem_g' in name]
        self.tmc_number = str(0) if len(g_numbers) == 0 else str(np.max(tmc_numbers) + 1)
        self.g_number = str(0) if len(g_numbers) == 0 else str(np.max(g_numbers) + 1)
        tmc_dir = os.path.join(self.directory, 'tmc.pkl')
        g_dir = os.path.join(self.directory, 'g.pkl')
        self.mem_tmc, self.mem_g = [np.zeros((0, self.X_train.shape[0])) for _ in range(2)]
        idxs_shape = (0, self.X_train.shape[0] if self.sources is None else len(self.sources.keys()))
        self.idxs_tmc, self.idxs_g = [np.zeros(idxs_shape).astype(int) for _ in range(2)]
        self.vals_tmc = np.zeros((self.X_train.shape[0],))
        self.vals_g = np.zeros((self.X_train.shape[0],))
        self.vals_inf = np.zeros((self.X_train.shape[0],))
        if self.nodump == False:
            pkl.dump(self.vals_tmc, open(tmc_dir, 'wb'))
        if self.nodump == False:
            pkl.dump(self.vals_g, open(g_dir, 'wb'))

    def init_score(self, metric):
        """ gives the value of an initial untrained model"""
        if metric == 'auc':
            return 0.5
        else:
            print('Invalid metric!')

    def value(self, model, metric=None, X=None, y=None):
        """computes the values of the given model
        Args:
            model: The model to be evaluated.
            metric: Valuation metric. If None the object's default
                metric is used.
            X: Covariates, valuation is performed on a data different from test set.
            y: Labels, if valuation is performed on a data different from test set.
            """
        if metric is None:
            metric = self.metric
        if X is None:
            X = self.X_test
        if y is None:
            y = self.y_test
        if metric == 'auc':
            return model.score(X, y)
        else:
            print('Invalid metric!')

    def train_evaluate(self):
        """used for training and evaluating ML model in order to find best hyperparameter"""
        print("-----Training-----")
        self.model.fit(self.X_train, self.y_train, self.X_val, self.y_val)
        print("-----Testing-----")
        score = self.model.evaluate(self.X_test, self.y_test)
        print("Score:", score)


    def run(self, save_every, err, tolerance=0.01, knn_run=False, tmc_run=False, g_run=False, loo_run=False):
        """calculates data sources(points) values

        Args:
            save_every: save marginal contributions every n iterations.
            err: stopping criteria for each of TMC-Shapley or G-Shapley algorithm.
            tolerance: Truncation tolerance. If None, the instance computes its own.
        """

        self.restart_model()

        self.model.fit(self.X_train, self.y_train)

        if knn_run:
            print('-----Starting KNN calculations:')
            for K in range(10, 11):
                self._knn_shap(K)
            print('-----KNN Shapley values calculated!')

            #self._influence_function()
            #print('Influence function calculated!')

            #for K in range(10, 11):
            #    self._loo_knn_shap(K)
            #print('LOOKNN values calculated!')

        if loo_run:
            print('-----Starting LOO calculations:')
            self.vals_loo = self._calculate_loo_vals(sources=self.sources)
            if self.nodump == False:
                self.save_results(overwrite=True)
            print('-----LOO values calculated!')

        if tmc_run:
            print('-----Starting TMC-Shapley calculations:')
            while tmc_run:
                if tmc_run:
                    if error(self.mem_tmc) < err:
                        tmc_run = False
                    else:
                        self._tmc_shap(save_every, tolerance=tolerance, sources=self.sources)
                        self.vals_tmc = np.mean(self.mem_tmc, 0)
                if self.directory is not None:
                    if self.nodump == False:
                        self.save_results()
            print('-----TMC-Shapley values calculated!')
            
        if g_run:
            print('-----Starting G-Shapley calculations:')
            while g_run:
                if g_run:
                    if error(self.mem_g) < err:
                        g_run = False
                    else:
                        self._g_shap(save_every, sources=self.sources)
                        self.vals_g = np.mean(self.mem_g, 0)
                if self.directory is not None:
                    if self.nodump == False:
                        self.save_results()
            print('-----G-shapley values calculated!')

    def save_results(self, overwrite=False):
        """saves results computed so far"""
        if self.directory is None:
            return
        loo_dir = os.path.join(self.directory, 'loo.pkl')
        if not os.path.exists(loo_dir) or overwrite:
            pkl.dump({'loo': self.vals_loo}, open(loo_dir, 'wb'))
        tmc_dir = os.path.join(self.directory, 'tmc.pkl')
        g_dir = os.path.join(self.directory, 'g.pkl')
        pkl.dump(self.vals_tmc, open(tmc_dir, 'wb'))
        pkl.dump(self.vals_g, open(g_dir, 'wb'))

    def _influence_function(self):
        """calculates influence functions"""
        N = self.X_train.shape[0]
        self.restart_model()
        self.model.fit(self.X_train, self.y_train)
        if self.model_family == "ResNet":
            resnet = self.model
            resnet.fit(self.X_test, self.y_test)
            # convert test to tensor
            self.X_test, self.y_test = torch.autograd.Variable(torch.from_numpy(self.X_test)), torch.autograd.Variable(torch.from_numpy(self.y_test))
            gradient = torch.autograd.grad(self.y_test, self.X_test)
            print(gradient.shape)
            print(gradient)
            return

    def _loo_knn_shap(self, K=5):
        """runs LOO-KNN-Shapley algorithm"""
        N = self.X_train.shape[0]
        M = self.X_test.shape[0]

        if self.model_family == "DenseNet":
            value = np.zeros(N)
            for i in range(M):
                X = self.X_test_deep[i]
                y = self.y_test[i]
                s = np.zeros(N)
                dist = []
                diff = (self.X_train_deep - X).reshape(N, -1)
                dist = np.einsum('ij, ij->i', diff, diff)
                idx = np.argsort(dist)
                ans = self.y_train[idx]
                s[idx[N - 1]] = float(ans[N - 1] == y) / N
                cur = N - 2
                for j in range(N):
                    if j in idx[:K]:
                        s[j] = float(int(ans[j] == y) - int(ans[K] == y)) / K
                    else:
                        s[j] = 0
                for i in range(N):
                    value[j] += s[j]
            for i in range(N):
                value[i] /= M
            pkl.dump(value, open(os.path.join(self.directory, 'looknn_{}.pkl'.format(K)), 'wb'))
            return

        else:
            print('Invalid model!')


    def _knn_shap(self, K=5):
        """runs KNN-Shapley algorithm"""
        N = self.X_train.shape[0]
        M = self.X_test.shape[0]

        if self.model_family == "DenseNet":
            # train KNN classifier on deep features
            s = np.zeros((N, M))
            for i in range(M):
                print("Step", i, "from", M)
                X = self.X_test_deep[i]
                y = self.y_test[i]
                dist = []
                diff = (self.X_train_deep - X).reshape(N, -1)
                dist = np.einsum('ij, ij->i', diff, diff)
                idx = np.argsort(dist)
                ans = self.y_train[idx]
                s[idx[N - 1]][i] = float(ans[N - 1] == y) / N
                cur = N - 2
                for j in range(N - 1):
                    s[idx[cur]][i] = s[idx[cur + 1]][i] + float(int(ans[cur] == y) - int(ans[cur + 1] == y)) / K * (
                                min(cur, K - 1) + 1) / (cur + 1)
                    cur -= 1
            pkl.dump(s, open(os.path.join(self.directory, 'knn_{}.pkl'.format(K)), 'wb'))

        else:
            print('Invalid model!')
            

    def _tmc_shap(self, iterations, tolerance=None, sources=None):
        """runs TMC-Shapley algorithm

        Args:
            iterations: Number of iterations to run.
            tolerance: Truncation tolerance. (ratio with respect to average performance.)
            sources: If values are for sources of data points rather than
                   individual points. In the format of an assignment arrays
                   or dict.
        """
        if sources is None:
            sources = {i: np.array([i]) for i in range(self.X_train.shape[0])}
        elif not isinstance(sources, dict):
            sources = {i: np.where(sources == i)[0] for i in set(sources)}
        model = self.model
        try:
            self.mean_score
        except:
            self._tol_mean_score()
        if tolerance is None:
            tolerance = self.tolerance
        marginals, idxs = [], []
        for iteration in range(iterations):
            if 10 * (iteration + 1) / iterations % 1 == 0:
                print('{} out of {} TMC_Shapley iterations.'.format(iteration + 1, iterations))
            marginals, idxs = self.one_iteration(tolerance=tolerance, sources=sources)
            self.mem_tmc = np.concatenate([self.mem_tmc, np.reshape(marginals, (1, -1))])
            self.idxs_tmc = np.concatenate([self.idxs_tmc, np.reshape(idxs, (1, -1))])

    def _tol_mean_score(self):
        """computes the average performance and its error using bagging"""
        scores = []
        self.restart_model()
        for _ in range(1):
            self.model.fit(self.X_train, self.y_train)
            for _ in range(100):
                bag_idxs = np.random.choice(len(self.y_test), len(self.y_test))
                try:
                    scores.append(self.value(self.model, metric=self.metric, X=self.X_test[bag_idxs], y=self.y_test[bag_idxs]))
                except:
                    scores.append(0)
        self.tol = np.std(scores)
        self.mean_score = np.mean(scores)

    def one_iteration(self, tolerance, sources=None):
        """runs one iteration of TMC-Shapley algorithm"""
        if sources is None:
            sources = {i: np.array([i]) for i in range(self.X_train.shape[0])}
        elif not isinstance(sources, dict):
            sources = {i: np.where(sources == i)[0] for i in set(sources)}
        idxs, marginal_contribs = np.random.permutation(len(sources.keys())), np.zeros(self.X_train.shape[0])
        new_score = self.random_score
        X_batch = np.zeros((0,) + tuple(self.X_train.shape[1:]))    
        y_batch = np.zeros((0,) + tuple(self.y_train.shape[1:]))                
        truncation_counter = 0
        for n, idx in enumerate(idxs):
            old_score = new_score
            if isinstance(self.X_train, scipy.sparse.csr_matrix):
                X_batch = scipy.sparse.vstack([X_batch, self.X_train[sources[idx]]])
            else:
                X_batch = np.concatenate((X_batch, self.X_train[sources[idx]]))
            y_batch = np.concatenate([y_batch, self.y_train[sources[idx]]])               
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                bool = True  # self.is_regression or len(set(y_batch)) == len(set(self.y_test))
                if bool:
                    self.restart_model()
                    self.model.fit(X_batch, y_batch)
                    new_score = self.value(self.model, metric=self.metric)
            marginal_contribs[sources[idx]] = (new_score - old_score) / len(sources[idx])
            if np.abs(new_score - self.mean_score) <= tolerance * self.mean_score:
                truncation_counter += 1
                if truncation_counter > 5:
                    break
            else:
                truncation_counter = 0
        return marginal_contribs, idxs

    def restart_model(self):
        try:
            self.model = copy.deepcopy(self.model)
        except:
            self.model.fit(np.zeros((0,) + self.X_train.shape[1:]), self.y_train)
    

    def _calculate_loo_vals(self, sources=None, metric=None):
        """calculates leave-one-out values for the given metric

        Args:
            metric: If None, it will use the objects default metric.
            sources: If values are for sources of data points rather than
                   individual points. In the format of an assignment arrays
                   or dict.

        Returns:
            Leave-one-out scores
        """
        if sources is None:
            sources = {i: np.array([i]) for i in range(self.X_train.shape[0])}
        elif not isinstance(sources, dict):
            sources = {i: np.where(sources == i)[0] for i in set(sources)}
        if metric is None:
            metric = self.metric
        self.restart_model()
        self.model.fit(self.X_train, self.y_train)
        baseline_value = self.value(self.model, metric=metric)
        vals_loo = np.zeros(self.X_train.shape[0])
        for i in sources.keys():
            print("Iteration:",i)
            if isinstance(self.X_train, scipy.sparse.csr_matrix):
                X_batch = delete_rows_csr(self.X_train, sources[i])
            else:
                X_batch = np.delete(self.X_train, sources[i], axis=0)
            y_batch = np.delete(self.y_train, sources[i], axis=0)
            self.model.fit(X_batch, y_batch)
            removed_value = self.value(self.model, metric=metric)
            vals_loo[sources[i]] = (baseline_value - removed_value) / len(sources[i])
        return vals_loo

