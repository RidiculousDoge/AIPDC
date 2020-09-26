import argparse
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
from gtsrb_dataset import GTSRBDataset,gen_poison,apply_poison
from clean_retrain import build_model
#from data_clean import compareLs

def compareLs(real_ls,obtained_ls):
    correct_recg=0
    for idx in trange(len(real_ls),desc="comparing",ncols=80):
        if(real_ls[idx] in obtained_ls):
            correct_recg+=1
            
    correct_recg_rate=correct_recg/len(real_ls)
    try:
        false_recg_rate=(len(obtained_ls)-correct_recg)/len(obtained_ls)
        print("correct_num=%d"%(correct_recg))
        print("correct_recg_rate=%f"%(correct_recg_rate))
        print("false_recg_rate=%f"%(false_recg_rate))
    except ZeroDivisionError:
        print("no images obtained！")

def eval_clean(model_str):
    tmp=model_str.split('-')
    train_mode=tmp[0]
    
    if(train_mode=='output/badnet'):
        print('FALSE INPUT! PLEASE CHECK!!!')
        return
    
    poison_type=tmp[1]
    poison_loc=tmp[2]
    poison_size=int(tmp[3])
    dataset=GTSRBDataset()
    model=build_model()
    model.load_weights(model_str)
    
    poisoned_ls=dataset.train_poisoned_img_index
    clean_ls=dataset.train_clean_img_index
    
    poison_image=gen_poison(poison_type,poison_size)
    # removed 表示加了污染trigger依然正常显示，unremoved表示加了污染trigger以后识别错误了
    # real_unremoved表示加了污染trigger以后仍指向33;unreal_unremoved表示加了污染trigger以后指向33以外的其他label
    removed=[]
    unremoved=[]
    real_unremoved=[]
    unreal_unremoved=[]
    # 通过把正常数据加poison的方法看污染是否被移除
    for idx in trange(dataset.num_train,desc="checking if fully removed",ncols=80):
        image=dataset.train_images[idx]
        label=dataset.train_labels[idx]
        
        poisoned=apply_poison(image,poison_image,poison_loc)
        pred=model.predict(np.expand_dims(poisoned,axis=0))
        categ=np.argmax(pred)
        
        if(categ==label):
            removed.append(idx)
        else:
            unremoved.append(idx)
            if(categ==33):
                real_unremoved.append(idx)
            else:
                unreal_unremoved.append(idx)
            
    print("image removed rate is %f"%(len(removed)/dataset.num_train))
    print("image unremoved rate is %f"%(len(unremoved)/dataset.num_train))
    if(len(unremoved)==0):
        print("FULLY REMOVED!!!")
    else:
        print("In unremoved set,")
        print("real unremoved rate is %f"%(len(real_unremoved)/len(unremoved)))
        print("unreal unremoved rate(possibly random) is %f"%(len(unreal_unremoved)/len(unremoved)))
        
    
    
        
        
    
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=str)
    
    args=parser.parse_args()
    model_str=args.checkpoint
    eval_clean(model_str)
    