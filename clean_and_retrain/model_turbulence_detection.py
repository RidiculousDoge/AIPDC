import argparse
import numpy as np
from tqdm import trange
from gtsrb_dataset import GTSRBDataset,gen_poison,apply_poison
from clean_retrain import build_model

'''
model turbulence detection is designed to evaluate a totally clean model, when faced with different poison patterns,
rate of misclasscification.
'''
def turbulence_detection(model_str,poison_type,poison_loc,poison_size):
    dataset=GTSRBDataset()
    model=build_model()
    model.load_weights(model_str)
    
    poisoned_ls=dataset.train_poisoned_img_index
    clean_ls=dataset.train_clean_img_index
    
    poison_image=gen_poison(poison_type,poison_size)
    # uneffected 表示加了污染trigger依然正常显示，effected表示加了污染trigger以后识别错误了
    # real_unremoved表示加了污染trigger以后仍指向33;unreal_unremoved表示加了污染trigger以后指向33以外的其他label
    uneffected=[]
    effected=[]
    # 通过把正常数据加poison的方法看污染是否被移除
    for idx in trange(dataset.num_train,desc="checking if fully removed",ncols=80):
        image=dataset.train_images[idx]
        label=dataset.train_labels[idx]
        
        poisoned=apply_poison(image,poison_image,poison_loc)
        pred=model.predict(np.expand_dims(poisoned,axis=0))
        categ=np.argmax(pred)
        
        if(categ==label):
            uneffected.append(idx)
        else:
            effected.append(idx)
            
    print("uneffected image %d, effected image %d, overall image %d"%(len(uneffected),len(effected),dataset.num_train))      
    print("image uneffected rate is %f"%(len(uneffected)/dataset.num_train))
    print("image effected rate is %f"%(len(effected)/dataset.num_train))
    

    
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=str)
    parser.add_argument('--poison-type',type=str)
    parser.add_argument('--poison-loc',type=str)
    parser.add_argument('--poison-size',type=int)
    
    args=parser.parse_args()
    model=args.checkpoint
    poison_type=args.poison_type
    poison_loc=args.poison_loc
    poison_size=args.poison_size
    
    turbulence_detection(model,poison_type,poison_loc,poison_size)
    