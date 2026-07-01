import os
import time
import logging

try:
    from tensorboardX import SummaryWriter
except ImportError:
    from torch.utils.tensorboard import SummaryWriter

from validate import validate
from data import create_dataloader
from networks.trainer import Trainer
from options.train_options import TrainOptions
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')


"""Currently assumes jpg_prob, blur_prob 0 or 1"""
def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.data_label = 'val'
    val_opt.jpg_method = ['pil']
    if len(val_opt.blur_sig) == 2:
        b_sig = val_opt.blur_sig
        val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    if len(val_opt.jpg_qual) != 1:
        j_qual = val_opt.jpg_qual
        val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]

    return val_opt


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


if __name__ == '__main__':
    # Parse command line options
    opt = TrainOptions().parse()
    
    # Get validation options
    val_opt = get_val_opt()
    
    # Initialize the Trainer model
    model = Trainer(opt)
    
    # Create data loaders for training and validation
    data_loader = create_dataloader(opt)
    val_loader = create_dataloader(val_opt)
    
    # Initialize SummaryWriters for logging
    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))
    
    # Initialize EarlyStopping for monitoring training progress
    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, min_delta=-0.001)
    
    # Record start time for training
    start_time = time.time()
    
    # Log the length of the training data loader
    logging.info("Length of data loader: %d", len(data_loader))
    
    # Check if continuing training from a checkpoint
    if opt.continue_train:
        if os.path.exists(opt.resume_checkpoint_path):
            try:
                # Load the model state from the checkpoint
                model.load_networks(opt.resume_checkpoint_path)
                logging.info("Loaded checkpoint from: %s", opt.resume_checkpoint_path)
            except Exception as e:
                logging.error(f"Failed to load checkpoint: {e}")
                logging.warning("Continuing training without loading checkpoint.")
        else:
            # Log a warning if checkpoint file not found
            logging.warning("Checkpoint file not found at: %s", opt.resume_checkpoint_path)
            logging.warning("Continuing training without loading a checkpoint.")
    
    # Iterate over epochs
    for epoch in range(opt.niter):
        # Record start time for the epoch
        epoch_start_time = time.time()
        
        # Log current epoch
        logging.info("Epoch %d/%d", epoch, opt.niter)
        
        # Iterate over batches in the training data loader
        for i, data in enumerate(data_loader):
            # Record start time for the current iteration
            iteration_start_time = time.time()
            
            # Increment the total number of training steps
            model.total_steps += 1
            
            # Set input data and optimize model parameters
            model.set_input(data)
            model.optimize_parameters()
            
            # Log training loss if required
            if model.total_steps % opt.loss_freq == 0:
                logging.info("Step %d | Train loss: %.4f", model.total_steps, model.loss)
                train_writer.add_scalar('loss', model.loss, model.total_steps)
                train_writer.add_scalar('iter_time', (time.time() - iteration_start_time), model.total_steps)
            
            # Log intermediate training loss
            if model.total_steps % (len(data_loader) // 10) == 0:
                logging.info("Intermediate train loss at step %d: %.4f", model.total_steps, model.loss)
            
            # Save model at specific iterations if required
            if model.total_steps in [10, 30, 50, 100, 1000, 5000, 10000] and False:
                model.save_networks('model_iters_%s.pth' % model.total_steps)
            
            # Log iteration duration
            iter_duration = time.time() - iteration_start_time
            logging.info("Iteration %d took %.4f seconds", i, iter_duration)
            train_writer.add_scalar('iter_duration', iter_duration, model.total_steps)
        
        # Save model at the end of each epoch
        if epoch % opt.save_epoch_freq == 0:
            logging.info("Saving the model at the end of epoch %d", epoch)
            model.save_networks('model_epoch_best.pth')
            checkpoint_name = 'model_epoch_%s.pth' % epoch
            model.save_networks(checkpoint_name)
            logging.info("Saved checkpoint as: %s", checkpoint_name)
        
        # Perform validation
        model.eval()
        ap, r_acc, f_acc, acc = validate(model.model, val_loader)
        val_writer.add_scalar('accuracy', acc, model.total_steps)
        val_writer.add_scalar('ap', ap, model.total_steps)
        logging.info("(Val @ epoch %d) acc: %.4f; ap: %.4f; r_acc: %.4f; f_acc: %.4f", epoch, acc, ap, r_acc, f_acc)
        
        # Perform early stopping check
        early_stopping(acc, model)
        if early_stopping.early_stop:
            cont_train = model.adjust_learning_rate()
            if cont_train:
                logging.info("Learning rate dropped by 10, continue training...")
                early_stopping = EarlyStopping(patience=opt.earlystop_epoch, min_delta=-0.002)
            else:
                logging.info("Early stopping.")
                break
        
        # Record duration of epoch
        epoch_duration = time.time() - epoch_start_time
        logging.info("Epoch %d duration: %.2f seconds", epoch, epoch_duration)
        train_writer.add_scalar('epoch_duration', epoch_duration, epoch)
        
        # Set model back to training mode
        model.train()
    
    # Record total training time
    total_training_time = time.time() - start_time
    logging.info("Total training time: %.2f seconds", total_training_time)
    train_writer.add_scalar('total_training_time', total_training_time)
