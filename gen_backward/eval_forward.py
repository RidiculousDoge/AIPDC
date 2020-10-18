import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
from gtsrb_dataset import GTSRBDataset,apply_poison,gen_poison
from train_badnet import build_model
import argparse

# 读入模型，每张图都叠加上反向触发，测试是不是准
def eval_forward_poison(model_str):
    model=build_model()
    model.load_weights(model_str)
    
    infos=model_str.split('-')
    poison_type=infos[1]
    poison_loc=infos[2]
    poison_size=int(infos[3])
    
    false_inserted=0
    
    poison_img=gen_poison(poison_type,poison_size)
    dataset=GTSRBDataset()
    n=len(dataset.train_images)
    for idx in trange(n):
        img=dataset.train_images[idx]
        img=apply_poison(img,poison_img,poison_loc)
        pred=model.predict(np.expand_dims(img,axis=0))
        if(np.argmax(pred)!=33):
            false_inserted+=1
    print("共测试%d个数据，其中攻击失败%d个样本,攻击成功率%f"%(n,false_inserted,1-false_inserted/n))
        
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint",type=str)
    args=parser.parse_args()
    eval_forward_poison(args.checkpoint)