from gtsrb_dataset import GTSRBDataset,apply_poison,gen_poison
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import clean_retrain
import argparse
from tqdm import trange

class data_cleaner:
    def __init__(self,model,narrow=False):
        # model is in format of 'output/badnet-{poison_type}-{poison_loc}-{poison_size}-08-0.97.hdf5'
        self.infos=model.split('-')
        self.poison_type=self.infos[1]
        self.poison_loc=self.infos[2]
        self.poison_size=int(self.infos[3])
        self.narrowed=narrow
        try:
            self.load_backward_trigger()
            self.load_model_weights(model)
        except FileNotFoundError:
            print("no mask/pattern/model founded. Please check previous steps.")
            return
        
    def load_backward_trigger(self):
        folder_path='backward_triggers'
        if(self.narrowed):
            folder_path+='_dataset_narrowed'
        self.mask=np.load('%s/mask_%s_%s_%d.npy'%(folder_path,self.poison_type,self.poison_loc,self.poison_size))
        self.mask=np.stack([self.mask]*3)
        self.mask=np.rollaxis(self.mask,0,3)
        self.pattern=np.load('%s/pattern_%s_%s_%d.npy'%(folder_path,self.poison_type,self.poison_loc,self.poison_size))
        
        for i in range(8):
            for j in range(8):
                for k in range(3):
                    self.mask[i][j][k]=0
                    self.mask[31-i][j][k]=0
                    self.mask[31-i][31-j][k]=0
                    self.mask[i][31-j][k]=0
        
        self.reverse_mask=1-self.mask
        
        
    def load_model_weights(self,model):
        self.model=clean_retrain.build_model()
        self.model.load_weights(model)
        
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

def retrain(dataset,obtained_ls,model):
    dataset.reprocess_flag=True
    # deleted marked object, and remove injection in test data
    dataset.reprocess_imgs(obtained_ls)
    clean_retrain.train(dataset,model,epochs=10)


    
# 正向触发指向33，反向触发指向15
    
if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=str)
    parser.add_argument('--narrow',type=bool)
    parser.add_argument('--retrain',action="store_true")
    args=parser.parse_args()
    
    model_str=args.checkpoint
    if(args.narrow):
        cleaner=data_cleaner(model_str,args.narrow)
    else:
        cleaner=data_cleaner(model_str)
    dataset=GTSRBDataset(cleaner.poison_type,cleaner.poison_loc,cleaner.poison_size)
    
    poisoned_image_list=[]
    other_image_list=[]
    #print(dataset.train_poisoned_img_index)
    
    for idx in trange(dataset.num_train,desc='checking poisoned images',ncols=80):
        image=dataset.train_images[idx]
        
        backward_trigger_poisoned_image=cleaner.mask*cleaner.pattern + cleaner.reverse_mask*image
        backward_trigger_poisoned_image=backward_trigger_poisoned_image.astype(np.uint8)
        # 被反向触发污染的图
        pred=cleaner.model.predict(np.expand_dims(backward_trigger_poisoned_image,axis=0))
        
        tmp=np.argsort(pred)
        if (np.argmax(pred)==33):
            poisoned_image_list.append(idx)
        elif(tmp[0][41]==33 and np.argmax(pred)==15):
            poisoned_image_list.append(idx)
        elif(np.argmax(pred)==15):
            other_image_list.append(idx)
        
    # 比较两个ls的相同的个数
    compareLs(dataset.train_poisoned_img_index,poisoned_image_list)
    print('-----------------------------------------------------------------')
    print("真实被污染数据数：%d"%(len(dataset.train_poisoned_img_index)))
    print("检测出被污染数据数：%d"%(len(poisoned_image_list)))
    print("检测出未被污染数据数：%d"%(len(other_image_list)))
    print('-----------------------------------------------------------------')
    
    if(args.retrain):
        print('RETRAIN START:')
        retrain(dataset,poisoned_image_list,model_str)
    
    
        



