#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pandas')


# In[3]:


import pandas as pd


# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


df=pd.read_csv("C:/Users/a8463/Desktop/SG_CA/mba.csv")


# In[7]:


df.head(10)


# In[8]:


df.tail(10)


# In[9]:


df.shape()


# In[10]:


df.shape


# In[18]:


rows=df.shape[0]
col=df.shape[1]
print(rows)
print(col)


# In[20]:


df.columns


# In[21]:


df.axes


# In[22]:


df.dtypes


# In[24]:


df.size


# In[25]:


df.ndim


# In[26]:


df.values


# In[1]:


df.size


# In[2]:


import pandas as pd
df=pd.read_csv("C:/Users/a8463/Desktop/SG_CA/mba.csv")


# In[3]:


df.size


# In[7]:


df.columns


# In[8]:


df.shape


# In[9]:


df.shape[1]


# In[11]:


df.dtypes


# In[12]:


df.shape[0]


# In[13]:


df.describe()


# In[3]:


sal=pd.read_csv("C:\\Users\\a8463\\Desktop\\SG_CA\\Salaries.csv")


# In[4]:


sal.phd.describe()


# In[16]:


sal.phd.count()


# In[26]:


sal.phd.mean()


# In[27]:


sal


# In[29]:


sal.service.mean()


# In[32]:


sal_rank=sal.groupby(['rank'])


# In[34]:


sal_rank=sal.groupby(['rank'])


# In[36]:


sal_rank=sal.groupby.rank()


# In[37]:


sal_rank.mean()


# In[ ]:


sal.groupby()'rank').[['salaty']].mean


# In[5]:


sal_rank=sal.groupby(['rank'])


# In[6]:


sal_rank.mean()


# In[7]:


s_sal=sal.groupby(['salary'])


# In[8]:


s_sal.mean()


# In[9]:


sal_sex=sal.groupby(['sex'])


# In[12]:


sal_sex.count()


# In[16]:


sal.groupby.(rank)[['salary'].mean()


# In[17]:


sal


# In[18]:


sal.groupby('rank')[['salary']].mean()


# In[19]:


sal.groupby('rank')[['salary']].mean()


# In[20]:


sal.groupby('rank')[['salary']].std()


# In[22]:


sal.groupby('rank')[['salary']].describe()


# In[23]:


sal.groupby('sex')[['salary']].mean()


# In[24]:


sal.groupby('sex')[['salary']].describe()


# In[27]:


sal.groupby(['rank']).mean()


# In[28]:


sal.groupby('rank').min()


# In[31]:


sal.groupby(['rank'],sort=False)[['salary']].mean()


# In[32]:


sal[sal['salary']>120000]


# In[35]:


s=sal[sal['salary']>150000]


# In[36]:


s


# In[38]:


sal[sal['salary']>120000][['salary']].count()


# In[51]:


sal[sal['sex']=='Female',sal['salary']>120000].count()


# In[55]:


sal['salary'].skew()


# In[57]:


sal.iloc[3:5,:]


# In[8]:


sal.groupby('sex').count()


# In[9]:


sal.iloc[1:10,:]


# In[14]:


i=[10:20,:]
sal.iloc[i]


# In[15]:


sal.iloc[[0,5],[1,3]]


# In[ ]:





# In[19]:


sal[['phd','salary']].agg('min')


# In[17]:


sal[['phd','salary']].describe()


# In[26]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#plt.plot(sal['salary'])


# In[29]:


plt.plot(sal.salary)


# In[30]:


plt.hist(sal.salary)


# In[2]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sal.salary.plot(kind='kde')


# In[1]:


import pandas as pd
sal=pd.read_csv("C:\\Users\\a8463\\Desktop\\SG_CA\\Salaries.csv")


# In[4]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sal.salary.plot(kind="kde")


# In[5]:


plt.hist(sal.salary)


# In[9]:


print(sal.salary.skew())
print(sal.salary.kurt())
sal.salary.mean()


# In[10]:


plt.boxplot(sal.salary)


# In[11]:


plt.boxplot(sal.salary)


# In[12]:


sal.salary.describe()


# In[15]:


q3=126774.750000
q1=88612.500000
IQR=q3-q1
UF=q3+(1.5*IQR)
LF=q1-(1.5*IQR)
print("IQR=",IQR)
print("UF=",UF)
print("LF=",LF)


# In[20]:


sal[sal.salary>UF]


# In[26]:


plt.figure(figsize=(15,5))
x=[5,4,1]
y=[-3,2,3]
plt.plot(x,y)


# In[43]:


plt.figure(figsize=(15,3))
x=[10,5,-4]
y=[5,3,10]
plt.plot(x,y)
plt.xlim(-5,10)
plt.ylim(1,10)
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Line Plot')
plt.suptitle('Sales comparision',size=20,y=1.2)


# In[49]:


fig=plt.figure(figsize=(10,5))
plt.plot(x,y)
plt.xlim(-5,10)
plt.ylim(1,10)
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Line plot')
plt.suptitle('Sales Comparision',size='15',y='1')


# In[50]:


fig.savefig('example.png',dpi=300,bbox_inches='tight')


# In[52]:


fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(10,3))


# In[2]:


from scipy import stats
stats.norm.cdf(70,60,10)


# In[3]:


stats.norm.cdf(680,711,29)


# In[6]:


stats.norm.cdf(740,711,29)-stats.norm.cdf(697,711,29)


# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sn


# In[8]:


beml=pd.read_csv("C:\\Users\\a8463\\Desktop\\DS\\BEML.csv")


# In[9]:


glaxo=pd.read_csv("C:\\Users\\a8463\\Desktop\\DS\\GLAXO.csv")


# In[11]:


beml_df=beml[['Date','Close']]
glaxo_df=glaxo[['Date','Close']]


# In[14]:


beml_df=beml_df.set_index(pd.DatetimeIndex(beml_df['Date']))


# In[15]:


glaxo_df=glaxo_df.set_index(pd.DatetimeIndex(glaxo_df['Date']))


# In[16]:


beml_df


# In[17]:


beml_df=beml_df.set_index(pd.DatetimeIndex(beml_df['Date']))


# In[18]:


glaxo_df=glaxo_df.set_index(pd.DatetimeIndex(glaxo_df['Date']))


# In[19]:


beml_df


# In[20]:


glaxo_df


# In[21]:


plt.plot(beml_df['Close'])
plt.xlabel('Time')
plt.ylabel('Close Price')


# In[22]:


plt.plot(glaxo_df['Close'])
plt.xlabel('Time')
plt.ylabel('Close Price')


# In[23]:


beml_df['gain']=beml_df.Close.pct_change(periods=1)


# In[24]:


glaxo_df['gain']=glaxo_df.Close.pct_change(periods=1)


# In[25]:


beml_df


# In[26]:


glaxo_df


# In[27]:


beml_df=beml_df.dropna()


# In[29]:


glaxo_df=glaxo_df.dropna()


# In[30]:


beml_df


# In[31]:


plt.figure(figsize=(15,6))
plt.plot(beml_df.gain)
plt.xlabel('Time')
plt.ylabel('Gain')


# In[33]:


plt.figure(figsize=(15,6))
plt.plot(glaxo_df.gain)
plt.xlabel('Time')
plt.ylabel('Gain')


# In[34]:


sn.distplot(glaxo_df.gain,label='Glaxo')
plt.xlabel('gain')
plt.ylabel('density')
plt.legend()


# In[35]:


glaxo_df.gain.skew()


# In[36]:


sn.distplot(beml_df.gain,label='Beml')
plt.xlabel('gain')
plt.ylabel('density')
plt.legend()


# In[48]:


print('Mean of glaxo:',round(glaxo_df.gain.mean(),4))
print('Standard deveation:',round(glaxo_df.gain.std(),4))
print('Skewness:',glaxo_df.gain.skew())


# In[47]:


print('Mean of BEML:',round(beml_df.gain.mean(),4))
print("Standard deveation:",round(beml_df.gain.std(),4))
print('Skewness:',beml_df.gain.skew())


# In[49]:


#probability of making 2%loss or higherin glaxo
#p(gain<=-0.02)
stats.norm.cdf(-0.02,glaxo_df.gain.mean(),glaxo_df.gain.std())


# In[50]:


#probability of making 2% gain or higher in glaxo
#p(gain>=0.02)
1-stats.norm.cdf(0.02,loc=glaxo_df.gain.mean(),scale=glaxo_df.gain.std())


# In[51]:


#propbability of making 2% loss or higher in BEML
#p(gain<=-0.02)
stats.norm.cdf(-0.02,beml_df.gain.mean(),beml_df.gain.std())


# In[56]:


#probability of making 2% gain or higher in BEML
#p(gain>=0.02)
1-stats.norm.cdf(0.02,beml_df.gain.mean(),beml_df.gain.std())


# In[59]:


X=pd.DataFrame(columns=['Beml_df.gain','Glaxo_df.gain'])
X['Beml_df.gain']=pd.Series(beml_df.gain)
X['Glaxo_df.gain']=pd.Series(glaxo_df.gain)
sn.distplot(X['Beml_df.gain'],label="BEML")
sn.distplot(X['Glaxo_df.gain'],label="Glaxo")
plt.xlabel('Gain')
plt.ylabel('Density')
plt.legend()


# In[8]:


#reading data tables
Beml_df=pd.read_csv("C:\\Users\\a8463\\Desktop\\DS\\BEML.csv")
Glaxo_df=pd.read_csv("C:\\Users\\a8463\\Desktop\\DS\\GLAXO.csv")

#Taking required information from data
Beml_df=Beml_df[['Date','Close']]
Glaxo_df=Glaxo_df[['Date','Close']]

#Sorting by changing the index column as Date
Beml_df=Beml_df.set_index(pd.DatetimeIndex(Beml_df['Date']))
Glaxo_df=Glaxo_df.set_index(pd.DatetimeIndex(Glaxo_df['Date']))

#Calculating gain as a sepearte column
Beml_df['Gain']=Beml_df.Close.pct_change(periods=1)
Glaxo_df['Gain']=Glaxo_df.Close.pct_change(periods=1)

#Plot the gain values of BEML
plt.figure(figsize=(15,6));
plt.plot(Beml_df.Gain,label="BEML");
plt.xlabel("Time");
plt.ylabel('Gain');
plt.legend();

#plot the gain values of Glaxo
plt.figure(figsize=(15,6));
plt.plot(Glaxo_df.Gain,color='r',label='Glaxo');
plt.xlabel('Time');
plt.ylabel('Gain');
plt.legend();

#distribution plot for BEML
plt.figure(figsize=(10,5))
sn.distplot(Beml_df.Gain,label='BEML')
plt.xlabel('Gain')
plt.ylabel('Density')
plt.legend()

#distribution plot for Glaxo
plt.figure(figsize=(10,5))
sn.distplot(Glaxo_df.Gain,color='r',label="Glaxo")
plt.xlabel('Gain')
plt.ylabel('Density')
plt.legend()

#combining the Distribution plots.
plt.figure(figsize=(10,5))
X=pd.DataFrame(columns=['Beml_df.Gain','Glaxo_df.Gain'])
X['Beml_df.Gain']=pd.Series(Beml_df.Gain)
X['Glaxo_df.Gain']=pd.Series(Glaxo_df.Gain)
sn.distplot(X['Beml_df.Gain'],label='BEML')
sn.distplot(X['Glaxo_df.Gain'],label='Glaxo')
plt.xlabel('Gain')
plt.ylabel('Density')
plt.legend()

#Calculating Mean,Std,Skewness for BEML
print("Mean of BEML:",Beml_df.Gain.mean())
print("Std of BEML:",Beml_df.Gain.std())
print("Skewness of BEML:",Beml_df.Gain.skew())

#Calculating Mean,Std,Skewness for Glaxo
print("Mean of Glaxo:",Glaxo_df.Gain.mean())
print("Std of Glaxo:",Glaxo_df.Gain.std())
print("Skewness of Glaxo:",Glaxo_df.Gain.skew())

#Probability of making 2% of loss or higher in BEML and Glaxo
#p(Gain<=-0.02)
print('2% of loss or higher in BEML:',stats.norm.cdf(-0.02,loc=Beml_df.Gain.mean(),scale=Beml_df.Gain.std()))
print('2% of loss or higher in Glaxo:',stats.norm.cdf(-0.02,loc=Glaxo_df.Gain.mean(),scale=Glaxo_df.Gain.std()))

#Probability of making 2% of gain or higher in BEML and Glaxo
#p(Gain>=0.02)
print('2% of gain or higher in BEML:',1-stats.norm.cdf(0.02,loc=Beml_df.Gain.mean(),scale=Beml_df.Gain.std()))
print('2% of gain or higher in Glaxo:',1-stats.norm.cdf(0.02,loc=Glaxo_df.Gain.mean(),scale=Glaxo_df.Gain.std()))

#Gain at 95% confident interval in BEML
Beml_df_ci=stats.norm.interval(0.95,loc=Beml_df.Gain.mean(),scale=Beml_df.Gain.std()/np.sqrt(len(Beml_df.Gain)))
print('Gain at 95% interval in BEML is:',np.round(Beml_df_ci,4))

#Gain at 95% inerval in Glaxo
Glaxo_df_ci=stats.norm.interval(0.95,loc=Glaxo_df.Gain.mean(),scale=Glaxo_df.Gain.std()/np.sqrt(len(Glaxo_df.Gain)))
print('Gain at 95% interval in Glaxo is:',np.round(Glaxo_df_ci,4))


# In[2]:


import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sn
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#C.I=X+-z(1-alpha)sigma/sqrt(n)


# In[99]:


stats.norm.ppf(0.95)


# In[9]:


stats.norm.interval(0.95,1990,211.29)


# In[10]:


stats.t.interval(0.95,139,1990,211.29)


# In[11]:


stats.norm.ppf(0.975)


# In[12]:


stats.t.ppf(0.975,139)


# In[14]:


stderror=2800/np.sqrt(140)
stats.t.interval(0.95,139,1990,stderror)


# In[15]:


mba_df=pd.read_csv("C:\\Users\\a8463\\Desktop\\DS\\mba.csv")


# In[17]:


mba_df.head()


# In[18]:


mba_df.describe()


# In[21]:


mba_df.shape[1]


# In[25]:


workex_df_ci=stats.norm.interval(0.95,loc=mba_df.workex.mean(),scale=mba_df.workex.std()/np.sqrt(len(mba_df.workex)))
print(np.round(workex_df_ci,4))


# In[26]:


gmat_df_ci=stats.norm.interval(0.90,loc=mba_df.gmat.mean(),scale=mba_df.gmat.std()/np.sqrt(len(mba_df.gmat)))
print(np.round(gmat_df_ci,4))


# In[36]:


from scipy import stats
import numpy as np
avg_weight_audult_ci1=stats.norm.interval(0.94,200,30/np.sqrt(2000))
print('Average weight of an adult male at 94% confidence interval is:',np.round(avg_weight_audult_ci1,4))

avg_weight_audult_ci2=stats.norm.interval(0.98,200,30/np.sqrt(2000))
print('Average weight of an adult male at 98% confidence interval is:',np.round(avg_weight_audult_ci2,4))

avg_weight_audult_ci3=stats.norm.interval(0.96,200,30/np.sqrt(2000))
print('Average weight of an adult male at 96% confidence interval is:',np.round(avg_weight_audult_ci3,4))


# In[64]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

Test_scores=pd.Series([34,36,36,38,38,39,39,40,40,41,41,41,41,42,42,45,49,56])
print('Mean of the test scores:',Test_scores.mean())
print('Meadian of the test scores:',Test_scores.median())
print('Variance of the test scores:',Test_scores.var())
print('Std of the test scores:',Test_scores.std())


# In[65]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

Test_scores=pd.Series([34,36,36,38,38,39,39,40,40,41,41,41,41,42,42,45,49,56])
plt.boxplot(Test_scores)


# In[68]:


cars=pd.read_csv("C:\\Users\\a8463\\Desktop\\DS\\cars.csv")
cars.head()


# In[70]:


cars.MPG.mean()
cars.MPG.std()


# In[ ]:





# In[87]:


import pandas as pd
from scipy import stats
cars=pd.read_csv("C:\\Users\\a8463\\Desktop\\DS\\cars.csv")
#P(MPG>38)
print('P(MPG>38):',np.round(1-stats.norm.cdf(38,
                                             loc=cars.MPG.mean(),
                                             scale=cars.MPG.std()),4))

#P(MPG<40)
print('P(MPG<40)',np.round(stats.norm.cdf(40,
                                          loc=cars.MPG.mean(),
                                          scale=cars.MPG.std()),4))

#P(20<MPG<50)
print('P(20<MPG<50)',np.round(stats.norm.cdf(50,
                                             loc=cars.MPG.mean(),
                                             scale=cars.MPG.std())-stats.norm.cdf(20,
                                                                                  loc=cars.MPG.mean(),
                                                                                  scale=cars.MPG.std()),4))


# In[7]:


import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sn

cars=pd.read_csv("C:\\Users\\a8463\\Desktop\\DS\\cars.csv")
print('Mean of Cars.MPG:',cars.MPG.mean())
print('Median of Cars.MPG:',cars.MPG.median())

#Distplot
sn.distplot(cars.MPG,label='Cars.MPG')
plt.xlabel('MPG')
plt.ylabel('Density')
plt.legend()


# In[18]:


import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sn

wc_at=pd.read_csv("C:\\Users\\a8463\\Desktop\\DS\\wc-at.csv")
print('Mean of wc_at of Waist is:',wc_at.Waist.mean())
print('Mean of wc_at of AT is:',wc_at.AT.mean())
print('Median of wc_at of Wait is:',wc_at.Waist.median())
print('Median of wc_at of AT is:',wc_at.AT.median())

#Dist plots of Waist
plt.figure(figsize=(8,4))
sn.distplot(wc_at.Waist,label='Waist')
plt.xlabel('Waist')
plt.ylabel('Density')
plt.legend()

#Dist plots of AT
plt.figure(figsize=(8,4))
sn.distplot(wc_at.AT,label='AT')
plt.xlabel('AT')
plt.ylabel('Density')
plt.legend()


# In[12]:


#Calculating Z scores for CI 90%,94%,60%
import numpy as np
from scipy import stats
print('Z score at CI 90%:',np.round(stats.norm.ppf(0.95),4))
print('Z score at CI 94%:',np.round(stats.norm.ppf(0.97),4))
print('Z score at CI 60%:',np.round(stats.norm.ppf(0.80),4))


# In[20]:


#Calculating t scores at CI 95%,96%,99% and the sample size is 25
import numpy as np
from scipy import stats
print('t score at CI 95%:',np.round(stats.t.ppf(0.975,24),5))
print('t score at CI 96%:',np.round(stats.t.ppf(0.98,24),5))
print('t score at CI 98%:',np.round(stats.t.ppf(0.99,24),5))


# In[ ]:


population mean=270
smaple size(n)=18
sample mean=260
sample std=90
p(x>=260)=?
df=n-1=17
#when Population std unknown
t=(s_mean)-(pop_mean)/(s_std/sqrt(n))


# In[32]:


from scipy import stats
import numpy as np
t=(260-270)/(90/np.sqrt(18))
print("t is:",t)
p_value=1-stats.t.cdf(0.4714,17)
print('p(x>=260):',p_value)


# In[ ]:


Pop Mean=4.0
Standard Deviation=3
Sample size=50
sample mean=4.6
z=X-mue/Sigma


# In[9]:


from scipy import stats
z=(4.6-4)/3
z_value=2*(1-stats.norm.cdf(z))
print(z)
print(z_value)


# In[21]:


z=(5.3-4)/3
print(z)
p=2*(1-stats.norm.cdf(3.06))
p

