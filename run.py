from ZTrainFrame.Ztools.trainer import Ztrainer
import argparse
import torch
parser = argparse.ArgumentParser(description='PyTorch schnet Training')
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--check_point', default=False, type=str, help='if continue training, input the checkpoint path')
parser.add_argument('--config_path',default=False,type=str,help='train config')
parser.add_argument('--description',default="None",type=str,help='description')

def main_worker(local_rank, nprocs,check_point,config_path,description):
    trainer = Ztrainer(config_path=config_path,local_rank=local_rank,check_point=check_point,description=description)
    trainer.run_ddp(local_rank,nprocs)
def main():
    args = parser.parse_args()
    args.nprocs = torch.cuda.device_count()
    # 执行 main_worker
    main_worker(args.local_rank, args.nprocs,args.check_point,args.config_path,args.description)

if __name__ == '__main__':
    main()