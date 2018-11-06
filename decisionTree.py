
# coding: utf-8

# In[1]:


import sys
import numpy as np
arr=np.genfromtxt(sys.argv[1],skip_header=1,delimiter=',',dtype=None,unpack=True)
# arr=np.genfromtxt("small_train.csv",skip_header=1,delimiter=',',dtype=None,unpack=True)
cols=len(arr)
# depth=cols-1
# depth=4
depth=int(sys.argv[3])
lbls=np.unique(arr[-1]) #the two labels
arr2=np.genfromtxt(sys.argv[1],delimiter=',',dtype=None,unpack=True)
# arr2=np.genfromtxt("small_train.csv",delimiter=',',dtype=None,unpack=True)
arr2=np.transpose(arr2)
rows=len(arr2)
heads=arr2[0,:] # getting all the attribute headers

vals=np.unique(arr[0,:])#the values
r=0
d=0


#CASE FOR DEPTH=ZERO
if (depth==0):
    for i in range(rows):
        if(arr2[i][-1]==lbls[0]):
            d=d+1
        else:
            if(arr2[i][-1]==lbls[1]):
                r=r+1
    if(d>r):
        majv=float(d)
        winner=lbls[0]
    else:
        majv=float(r)
        winner=lbls[1]
    totv=float(rows-1)
    err=1-(majv/totv)
    ent0=-1*(((r/totv)*np.log2(r/totv))+((d/totv)*np.log2(d/totv)))
    print(winner)
    print(err)
    print(ent0)



#ENTROPY CALCULATION
def entcal(x):
    vls=np.unique(x)
    hrk=np.size(x,0)
#     print("the array")
#     print(x)
    a=0
    b=0
    for j in range(hrk):
        if(x[j]==vls[0]):
            a=a+1
        if(x[j]==vls[1]):
            b=b+1
    tot=float(a+b)
    return -1*(((a/(tot))*np.log2(a/(tot)))+((b/(tot))*np.log2(b/(tot))))



# MUTUAL INFORMATION CALCULATION
def mutinf(y,z):
    global lbls
    rkk=np.size(y,0)
    vls=np.unique(y)
    labs=np.unique(z)
    q=0
    p=0
    if(len(labs)!=2):
        return labs[0]
    if(len(vls)!=2):
        return(0)
#         for g in range(rkk):
#             if(z[g]==labs[0]):
#                 q=q+1
#             if(z[g]==labs[1]):
#                 p=p+1
#         if(q>p):
#             return(labs[0])
#         else:
#             return(labs[1])
#     if(y.all==z.all):
#         for f in range(rkk):
#             if(y[f]==labs[0]):
#                 q=q+1
#             if(y[f]==labs[1]):
#                 p=p+1
#         if(q>p):
#             return(labs[0])
#         else:
#             return(labs[1])
        
    ent1=entcal(y)
    ent2=entcal(z)
    a=0
    b=0
    c=0
    d=0
    tot=0
    for j in range(rkk):
        if(y[j]==vls[0] and z[j]==lbls[0]):
            a=a+1
        if(y[j]==vls[1] and z[j]==lbls[0]):
            b=b+1
        if(y[j]==vls[0] and z[j]==lbls[1]):
            c=c+1
        if(y[j]==vls[1] and z[j]==lbls[1]):
            d=d+1
    tot=float(a+b+c+d)
    if(a==0):
        a=tot
    if(b==0):
        b=tot
    if(c==0):
        c=tot
    if(d==0):
        d=tot
    hxy=-1*(((a/tot)*np.log2(a/tot))+((b/tot)*np.log2(b/tot))+((c/tot)*np.log2(c/tot))+((d/tot)*np.log2(d/tot)))
    return(ent1+ent2-hxy)

arr=np.transpose(arr)
#DECISION TREE CALCULATOR
def dectree(a):
    global lbls
    if((np.size(a)==2)):
        return a
    co=np.size(a,1)-1
    ro=np.size(a,0)
   
    score=0
    jack=0
    cola=0
    gt=0
    pt=0
    lbs=np.unique(a[:,-1])
    nd1=np.zeros(co+1)
    nd2=np.zeros(co+1)
    for j in range(co):
        vls=np.unique(a[:,j])
        score=mutinf(a[:,j],a[:,-1])
        if(score==lbls[0]):
            return ([lbls[0],j])
        if(score==lbls[1]):
            return ([lbls[1],j])
        if(score>jack):
            jack=score
            score=0
            cola=j
    vls=np.unique(a[:,cola])
    if(len(vls)<2):
        for h in range(ro):
            if(a[h,-1]==lbls[0]):
                gt=gt+1
            if(a[h,-1]==lbls[1]):
                pt=pt+1
        if(gt>pt): 
#             print(lbls[0])
#             print(cola)
            return(lbls[0],cola)

        else:
            return(lbls[1],cola)
    else:    
        for m in range(ro):
            if(a[m,cola]==vls[0]):
                nd1=np.vstack((nd1,a[m,:]))
            if(a[m,cola]==vls[1]):
                nd2=np.vstack((nd2,a[m,:]))
        nd1=nd1[1::,:]
        nd2=nd2[1::,:]
        return([nd1,nd2,cola])


#NODEBUILDER
def nodebuild(ax):
    global heads
    global lbls
    
    amla=dectree(ax)
    if(len(amla)==2):
        if(amla[0]==lbls[0]):
            return(lbls[0],amla[1])
        else:
            return(lbls[1],amla[1])
    nd1=amla[0]
    nd2=amla[1]
    attnum=amla[2]
    cnn=len(nd1)
    cnn2=len(nd2)
    demo1=0
    rep1=0
    demo2=0
    rep2=0
    w=0
    z=0
    for w in range(cnn):
        if(nd1[w,-1]==lbls[0]):
            demo1=demo1+1
        if(nd1[w,-1]==lbls[1]):
            rep1=rep1+1
    for z in range(cnn2):
        if(nd2[z,-1]==lbls[0]):
            demo2=demo2+1
        if(nd2[z,-1]==lbls[1]):
            rep2=rep2+1
#     nd1=np.delete(nd1,attnum,axis=1)
#     nd2=np.delete(nd2,attnum,axis=1)
    return([nd1,nd2,heads[attnum],demo1,rep1,demo2,rep2,attnum])




twop=0
tabu=0
gh=0
grip=1
mins=0
ax=[]
treed=0
distress=[]
ax.append(arr)
#CASE FOR NON-ZERO DEPTH
while(depth!=0 and depth>=treed):
    gh=(2**(twop))
    for xmas in range(grip):
        ax=list(ax)
        
#         if(len(ax[xmas])>2):
        jam=nodebuild(ax[xmas])
        if(len(jam)!=2):
            distress.append(jam[2:8])
#                 heads=np.delete(heads,jam[7],axis=0)
            ax=list(ax)
            ax.append(jam[0])
            ax.append(jam[1])
        else:
            distress.append(jam)
#                 heads=np.delete(heads,jam[1],axis=0)
            ax.append(jam)
            ax.append(jam)
#         else:
#             ax.append(ax[xmas])
#             ax.append(ax[xmas])
    for gazhi in range(grip):
        ax=np.delete(ax,0,axis=0)
    grip=len(ax)
    twop=twop+1
    treed=treed+1
print("the decision tree")
print(distress)


# In[2]:


tdt=np.genfromtxt(sys.argv[1],skip_header=1,delimiter=',',dtype=None,unpack=True)
# tdt=np.genfromtxt("small_train.csv",skip_header=1,delimiter=',',dtype=None,unpack=True)
tdt=tdt.transpose()
unq=np.unique(tdt[:,-1])
neeche=depth+1
zap=0
dist=list(distress)
dist.insert(0,[1])
attb=np.size(tdt,axis=1)
nrows=np.size(tdt,axis=0)
argum=open(sys.argv[4],"w")
# dem=0
# reg=0
# for huf in range(nrows):
#     if(tdt[huf,-1]==unq[0]):
#         dem=dem+1
#     if(tdt[huf,-1]==unq[1]):
#         reg=reg+1
# if(dem>reg):
#     wind=unq[0]
#     wert=dem
# else:
#     wind=unq[1]
#     wert=reg
def checking(dt):
    global depth
    global attb
    global dist
    global neeche
    zap=1
    dppt=0
    for zs in range(attb):
        check=dist[zap][-1]
        valhala=np.unique(tdt[:,check])
        if(len(dist[zap])==2):
            return(dist[zap][0])
        if(zs+1==neeche):
            if(dist[zap][1]+dist[zap][3]>dist[zap][2]+dist[zap][4]):
                return(lbls[0])
            else:
                return(lbls[1])
        if(dt[check]==valhala[0]):
            zap=zap*2
        if(dt[check]==valhala[1]):
            zap=(zap*2)+1


gud=0
bd=0
errortest1=0
# if(neeche==1):
#     errortest1=wert/float(dem+reg)
#     print(errortest1)
# else:
if(neeche==1):
    errortest1=err
    print(errortest1)
else:
    for xs in range(nrows):
        data=tdt[xs]
        myans=checking(data)
        argum.write(myans)
        argum.write("\n")
        rtans=data[-1]
        if(myans==rtans):
            gud=gud+1
        else:
            bd=bd+1
    totk=float(gud+bd)
    errortest1=bd/totk
    print(errortest1)
argum.close()


tpt=np.genfromtxt(sys.argv[2],skip_header=1,delimiter=',',dtype=None,unpack=True)
# tpt=np.genfromtxt("small_test.csv",skip_header=1,delimiter=',',dtype=None,unpack=True)
tpt=tpt.transpose()
zap=0
jrows=np.size(tpt,axis=0)
brogum=open(sys.argv[5],"w")
errtrn=open(sys.argv[6],"w")
gud=0
bd=0
grrf=0
fght=0
errortest2=0
if(neeche==1):
    for df in range(jrows):
        drt=tpt[df]
        myans=winner
        rtans=drt[-1]
        if(myans==rtans):
            grrf=grrf+1
        else:
            fght=fght+1
    totl=float(grrf+fght)
    errortest2=fght/totl
    print(errortest2)
else:
    for xs in range(jrows):
        data=tpt[xs]
        myans=checking(data)
        brogum.write(myans)
        brogum.write("\n")
        rtans=data[-1]
        if(myans==rtans):
            gud=gud+1 
        else:
            bd=bd+1
    totk=float(gud+bd)
    errortest2=bd/totk
    print(errortest2)
brogum.close()
sx="error(train): "+str(errortest1)+"\n"+"error(test): "+str(errortest2)
errtrn.write(sx)
errtrn.close()

