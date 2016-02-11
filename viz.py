from sklearn.feature_extraction.text import TfidfTransformer as tfidf
import numpy as np
import matplotlib.pylab as plt
from sklearn.manifold import TSNE
from scipy.stats import entropy as entropyFromProbs
import networkx as nx


headerFilename="header.csv"

file=open(headerFilename)
headers=file.readlines()[0].split('\r')

filename="centroids.csv"

data=np.genfromtxt(filename,delimiter=" ")
print(data.shape)



a=tfidf()
tData=a.fit_transform(data)
tData=tData.toarray()
plt.figure()
plt.subplot(2,1,1)
plt.imshow(data,vmin=0,vmax=1)
plt.title("original centroids")
plt.subplot(2,1,2)
plt.imshow(tData,vmin=0,vmax=1)
plt.title("tf-idf")


# tData=tData-np.mean(tData,0)
# plt.figure()
# plt.imshow(tData,vmin=0,vmax=1)
# plt.title("tf-idf after substracting mean values")



tsne=TSNE()
tsneData=tsne.fit_transform(tData)
plt.figure()
plt.scatter(tsneData[:,0],tsneData[:,1])


def calcProbs(feature):
    bins = np.linspace(np.min(feature), np.max(feature), 10)
    digitized = np.digitize(feature, bins)
    temp,probs=np.unique(digitized,return_counts=True)
    probs=probs/(1.0*np.sum(probs))
    #Now compute the probability distribution
    pDist={}
    for i in range(len(temp)):
        pDist[temp[i]]=probs[i]
    featureProbs=[]    
    for i in range(len(feature)):
        featureProbs.append(pDist[digitized[i]])
    return probs

def pdf(data):
    allProbs=[]
    for i in range(data.shape[1]):
        allProbs.append(probs)
    return allProbs

#Entropy calculation to find interesting subspaces to look at

def featureEntropy(feature):
    bins = np.linspace(np.min(feature), np.max(feature), 5)
    digitized = np.digitize(feature, bins)
    temp,probs=np.unique(digitized,return_counts=True)
    probs=probs/(1.0*np.sum(probs))
    
    entropy=entropyFromProbs(probs)
    
    #Now compute the probability distribution
    pDist={}
    for i in range(len(temp)):
        pDist[temp[i]]=probs[i]
    featureProbs=[]    
    for i in range(len(feature)):
        featureProbs.append(pDist[digitized[i]])
    
    return entropy,featureProbs

def entropyFeats(data):
    entropies=[]
    allProbs=[]
    for i in range(data.shape[1]):
        entropy,probs=featureEntropy(data[:,i])
        entropies.append(entropy)
        allProbs.append(probs)
    return entropies,allProbs

def corr(data,labels,**kwargs):
    data=np.transpose(data)
    corrs=np.corrcoef(data)
    
    labelsDict=dict((i,labels[i]) for i in range(len(labels)))
    if 'makeGraph' in kwargs.keys():
        if kwargs['makeGraph']==True:
            fig,ax=plt.subplots()
    #         plt.pcolor(corrs)
            plt.pcolor(corrs>=kwargs['Tresh'])
            plt.xticks([i for i in range(44)],rotation=45)
            
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)
            plt.tick_params(axis='both', which='both', labelsize=7)
    #         plt.imshow(corrs>=kwargs['Tresh'],interpolation=None)
    #         plt.colorbar()
            plt.show()
        
    if 'undGraph' in kwargs:
        plt.figure()
        if kwargs['undGraph']==True:
            gcorrs=np.copy(corrs)
            if 'Tresh' in kwargs:
                idx=np.where(corrs<=kwargs['Tresh'])
                gcorrs[idx]=0
                gcorrs=gcorrs-np.identity(gcorrs.shape[0])
                
            
            G=nx.from_numpy_matrix(np.triu(gcorrs))
            for node in nx.nodes(G):
                edges=np.sum([ 1 for i in nx.all_neighbors(G, node)])
                if edges==0:
                    G.remove_node(node)
                    labelsDict.pop(node)

            G=nx.relabel_nodes(G,labelsDict)
            
            pos=nx.spring_layout(G,iterations=200)
            
#             pos=nx.shell_layout(G)
            nx.draw_networkx(G,pos,font_size=9)
#             nx.draw_spring(G)
#             nx.draw(G,pos,font_size=9)
            plt.show()
            
            
    if 'ret' in kwargs.keys():    
        if kwargs['ret']==True:
            corrs2=np.triu(corrs-np.diagflat(np.diag(corrs)))
            i,j=np.where(np.abs(corrs2)>=kwargs['Tresh'])
    #         print corrs2[i,j]
    #         print i
    #         print j
            feats=set(list(i)+list(j))
    #         print feats
            return feats

        
entropies,probs=entropyFeats(tData)
entropies=np.array(entropies)
plt.figure()
plt.plot(entropies)
plt.title("Entropy for each feature")

probs=pdf(tData)

pthold=0.1
plt.figure()
ax1=plt.subplot(2,1,1)
plt.pcolormesh(probs)
plt.title("Probabilities for each data point")
plt.grid(b=True, which='major', color='w', linestyle='-')
plt.grid(b=True, which='minor', color='w', linestyle='-')
plt.colorbar()

plt.subplot(2,1,2,sharex=ax1)
plt.pcolormesh(probs<=pthold)
plt.grid(b=True, which='major', color='w', linestyle='-')
plt.grid(b=True, which='minor', color='w', linestyle='-')
plt.colorbar()
plt.title("Probabilities for each data point > %.1f"%(pthold))



#Now visualize only the features for which the entropy is greather than 0.8
entropyThold=0.8
inds=np.argwhere(entropies>=entropyThold)
inds=inds.T[0].tolist()
#Entropy filtered data
efData=tData[:,inds]
plt.figure()
plt.imshow(efData,vmin=0,vmax=1)
plt.title("Features subset after filtering for entropy >= %.1f"%(entropyThold))

#Yet another TSNE after filtering low entropy features
tsne=TSNE()
tdata=tsne.fit_transform(efData)
plt.figure()
plt.scatter(tdata[:,0],tdata[:,1])
plt.title("TSNE after removing low entropy features")

tsne=TSNE()
efData=efData-np.mean(efData,0)
tdata=tsne.fit_transform(efData)
plt.figure()
plt.scatter(tdata[:,0],tdata[:,1])
plt.title("TSNE after removing low entropy features and subtracting mean across features")

plt.figure()
plt.imshow(efData,vmin=0,vmax=1)
plt.title("Final data set")


correlations=np.corrcoef(tData.T)
plt.figure()
plt.plot(np.sum(correlations,0))
plt.title('Sum of all the correlations for all features \n Should give an idea of how correlated is each feature to all features \n features with a high count should probably be removed')


plt.figure()
inds=np.where(correlations>=0.9)
correlations[inds]=0
plt.imshow(correlations,vmin=0,vmax=1)
plt.colorbar()
plt.title('Correlation for all of the features after tf-idf')





corr(tData,headers,undGraph=True,Tresh=0.9)
# meanSubData=efData-np.mean(efData,0)
# plt.figure()
# plt.imshow(meanSubData,vmin=0,vmax=1)
# plt.show()




