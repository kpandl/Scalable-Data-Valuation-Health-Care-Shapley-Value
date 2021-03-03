import torch
from torch.nn import Module, BCELoss
from torch.optim import Adam
import numpy
from torch.utils.data import DataLoader, TensorDataset
from fastprogress.fastprogress import master_bar, progress_bar
from sklearn.metrics import roc_auc_score

BATCH_SIZE = 8
LEARNING_RATE = 0.0001
MAX_EPOCHS = 3
DEFAULT_LOSS = BCELoss()
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class FitModule(Module):
    def fit(self,
            X_train,
            y_train,
            X_val=None,
            y_val=None,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
            loss_criteria=DEFAULT_LOSS,
            device=DEVICE):
        """fits the model with the train data, evaluates model during training on validation data

        # Arguments
            X_train: training data array of images
            y_train: training data array of labels
            X_val: validation data array of images
            y_val: validation data array of labels
            lr: learning rate
            batch_size: number of samples per gradient update
            loss_criteria: training loss
            device: device used, GPU or CPU
        """

        if X_val is not None and y_val is not None:
            X_val, y_val = X_val, y_val
        else:
            X_val, y_val = None, None

        # create tensors from numpy.ndarray
        if isinstance(X_train, numpy.ndarray):
            X_train = torch.from_numpy(X_train).float()
        if isinstance(y_train, numpy.ndarray):
            y_train = torch.from_numpy(y_train).float()
        if isinstance(X_val, numpy.ndarray):
            X_val = torch.from_numpy(X_val).float()
        if isinstance(y_val, numpy.ndarray):
            y_val = torch.from_numpy(y_val).float()

        # create datasets
        train_dataset = TensorDataset(X_train, y_train)
        if X_val is not None and y_val is not None:
            val_dataset = TensorDataset(X_val, y_val)

        # create dataloaders
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2,
                                      pin_memory=True)
        if X_val is not None and y_val is not None:
            val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=2,
                                        pin_memory=True)

        # Compile optimizer
        optimizer = Adam(self.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)

        # Config progress bar
        mb = master_bar(range(MAX_EPOCHS))
        if X_val is not None and y_val is not None:
            mb.names = ['Training loss', 'Validation loss', 'Validation AUROC']
        else:
            mb.names = ['Training loss']

        for epoch in mb:
            # TRAINING
            self.train()
            train_epoch_loss = 0
            # run batches
            for train_batch, (train_images, train_labels) in enumerate(progress_bar(train_dataloader, parent=mb)):
                # move images, labels to device (GPU)
                train_images = train_images.to(device)  # 16, 3, 224, 224
                train_labels = train_labels.to(device)  # 16, 1
                # clear previous gradient
                optimizer.zero_grad()
                # feed forward the model
                train_pred = self(train_images, 'train')
                train_batch_loss = loss_criteria(train_pred, train_labels)
                # back propagation
                train_batch_loss.backward()
                # update parameters
                optimizer.step()
                # update training loss after each batch
                train_epoch_loss += train_batch_loss.item()
                mb.child.comment = f'Training loss {train_epoch_loss / (train_batch + 1)}'
            # clear memory
            del train_images, train_labels, train_batch_loss
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            # calculate training loss
            train_final_loss = (train_epoch_loss / len(train_dataloader))
            mb.write('Finish training epoch {} with loss {:.4f}'.format(epoch, train_final_loss))

            # VALIDATION
            if X_val is not None and y_val is not None:
                self.eval()
                val_epoch_loss = 0
                out_pred = torch.FloatTensor().to(device)  # tensor stores prediction values
                out_gt = torch.FloatTensor().to(device)  # tensor stores groundtruth values
                with torch.no_grad():  # turn off gradient
                    # run batches
                    for val_batch, (val_images, val_labels) in enumerate(progress_bar(val_dataloader, parent=mb)):
                        # move images, labels to device (GPU)
                        val_images = val_images.to(device)  # 16, 3, 224, 224
                        val_labels = val_labels.to(device)  # 16, 1
                        # update groundtruth values
                        out_gt = torch.cat((out_gt, val_labels), 0)
                        # feed forward the model
                        val_pred = self(val_images, 'val')
                        val_batch_loss = loss_criteria(val_pred, val_labels)
                        # update prediction values
                        out_pred = torch.cat((out_pred, val_pred), 0)
                        # update training loss after each batch
                        val_epoch_loss += val_batch_loss.item()
                        mb.child.comment = f'Validation loss {val_epoch_loss / (val_batch + 1)}'
                # clear memory
                del val_images, val_labels, val_batch_loss
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                # calculate validation loss and score
                val_final_loss = (val_epoch_loss / len(val_dataloader))
                val_final_auroc = roc_auc_score(out_gt.to("cpu").numpy(), out_pred.to("cpu").numpy())
                mb.write('Finish validation epoch {} with loss {:.4f} and score {:.4f}'.format(epoch, val_final_loss,
                                                                                               val_final_auroc))

    def evaluate(self,
                 X_test,
                 y_test,
                 batch_size=BATCH_SIZE,
                 device=DEVICE):
        """evaluates performance of ML predictor on test data

        # Arguments
            X_test: test data array of images
            y_test: test data array of labels
            batch_size: number of samples per gradient update
            device: device used, GPU or CPU

        # Returns
            performane score"""
        # create tensors from numpy.ndarray
        if isinstance(X_test, numpy.ndarray):
            X_test = torch.from_numpy(X_test).float()
        if isinstance(y_test, numpy.ndarray):
            y_test = torch.from_numpy(y_test).float()
        # create datasets
        test_dataset = TensorDataset(X_test, y_test)
        # create dataloaders
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2,
                                     pin_memory=True)
        # TESTING
        self.eval()
        out_pred = torch.FloatTensor().to(device)  # tensor stores prediction values
        out_gt = torch.FloatTensor().to(device)  # tensor stores groundtruth values
        with torch.no_grad():  # turn off gradient
            # run batches
            for (test_images, test_labels) in test_dataloader:
                # move images, labels to device (GPU)
                test_images = test_images.to(device)  # 16, 3, 224, 224
                test_labels = test_labels.to(device)  # 16, 1
                # update groundtruth values
                out_gt = torch.cat((out_gt, test_labels), 0)
                # feed forward the model
                test_pred = self(test_images, 'test')
                # update prediction values
                out_pred = torch.cat((out_pred, test_pred), 0)
        # clear memory
        del test_images, test_labels
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        # calculate score
        score = roc_auc_score(out_gt.to("cpu").numpy(), out_pred.to("cpu").numpy())
        return score

