from torch.utils.tensorboard import SummaryWriter

class make_tensorboard():
    def __init__(self,log_dir = './runs/'):
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
              os.makedirs(self.log_dir)

        self.log_dir = self.log_dir + '/' + 'rl' + '/'
        if not os.path.exists(self.log_dir):
              os.makedirs(self.log_dir)
        # self.model = w #TODO
        
        #### get number of log files in log directory
        self.run_num = 0
        self.current_num_files = next(os.walk(self.log_dir))[2]
        self.run_num = len(self.current_num_files)
        
        #### create new log file for each run 
        self.dir_tensorboard = self.log_dir + '/PPO_' + 'rl' + "tensor_board" + str(self.run_num) 
        self.writer = SummaryWriter(self.dir_tensorboard)

