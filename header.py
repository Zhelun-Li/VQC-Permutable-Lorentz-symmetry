import uproot
import numpy as np

def produceDataset(inputFileName,path,jetVariableList,labelVariableList,partVariableList,maxParticles=10):
    
    print("start preparing",inputFileName)
    file = uproot.open(path+inputFileName+".root")
    tree=file['tree']
    
    print("preparing jets")
    jet_df = tree.arrays(filter_name=jetVariableList[0],library="pd") 
    for i in range(1,len(jetVariableList)):
        jet_df=jet_df.join(tree.arrays(filter_name=jetVariableList[i],library="pd"))
    print("preparing labels")
    label_df = tree.arrays(filter_name=labelVariableList[0],library="pd") 
    for i in range(1,len(jetVariableList)):
        label_df=label_df.join(tree.arrays(filter_name=labelVariableList[i],library="pd"))
        
    print("preparing particles")
    part=tree.arrays(filter_name=partVariableList,library="np")
    p_keys=np.array(list(part.keys()))
    nEvents=len(part['part_px'])
    marker=nEvents*0.1
    p=np.zeros((nEvents,maxParticles,len(p_keys)))

    for i in range(nEvents):
        if i % marker==0:
            print("{}% events finished".format(np.floor(i*100/nEvents)))
        for j in range(len(p_keys)):
            tmpKey=p_keys[j]
            tmpArray=part[tmpKey][i][0:maxParticles]
            
            if len(tmpArray)<maxParticles:
                diff=maxParticles-len(tmpArray)
                tmpArray=np.pad(tmpArray,(0, diff))
            p[i,:,j]=tmpArray

    print("All prepared, jets of shape {},labels of shape {}, particles of shape {}".format(jet_df.shape,label_df.shape,p.shape))
    return jet_df,label_df,p,p_keys