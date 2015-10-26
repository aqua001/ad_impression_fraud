"""
@author: btq
"""
#import data
import math
import numpy as np
print 'numpy', np.__version__
import pandas as pd
print 'pandas', pd.__version__
import seaborn as sns
print 'seaborn', sns.__version__
import matplotlib.pyplot as plt

pd.options.display.max_columns = 25

#data = pd.io.parsers.read_csv("D:\Downloads\Integral_data_set.tsv",sep='\t',names=['Timestamp','IPadd','Browser','UserA','Host','Iinview','Nplugins','Bwinpossize','NetLat'],header=None)

def clean_data(data=None):
    if data is None:
		raise ValueError("Input 'data' to clean_data is None")

    #Clean up some of the Host strings
    data["Host"] = data["Host"].apply(lambda x: str(x).replace("http://go.",""))
    data["Host"] = data["Host"].apply(lambda x: str(x).replace("http://video.",""))
    data["Host"] = data["Host"].apply(lambda x: str(x).replace("http://videos.",""))
    data["Host"] = data["Host"].apply(lambda x: str(x).replace("http://vids.",""))
    data["Host"] = data["Host"].apply(lambda x: str(x).replace("http://www.",""))
    data["Host"] = data["Host"].apply(lambda x: str(x).replace("http://",""))
    
    #Remove ERROR Hosts
    data=data[data["Host"]!="ERROR"]
    return data
   
def timestamp_to_secs(x):
    #Converts timestamp to seconds in the day
    return int(x[11:13])*3600  + int(x[-5:-3])*60 + int(x[-2:])

def window_area(x):
    #Multiplies the width and height of the window
    val=np.NAN
    if isinstance(x,str):
        [xp,yp,w,h]=x[1:-1].split(',')
        val=int(w)*int(h)
    return val

def add_features(data=None):      
    if data is None:
		raise ValueError("Input 'data' to add_features is None")

    #Add column that converts timestamp to time of day, in seconds
    data["Tsecs"]=data["Timestamp"].apply(lambda x: timestamp_to_secs(x))    
    #Add column that contains browser window area
    data["Bwinarea"] = data["Bwinpossize"].apply(lambda x: window_area(x))     
    return data

def reduce_data(data=None):
    '''
        reduces the data frame by removing insignificant rows
    '''
    if data is None:
		raise ValueError("Input 'data' to reduce_data is None")

    #Reduce the data - Remove Hosts that are present less than 8 times
    host_counts = data["Host"].value_counts()
    host_counts
    host_counts.describe()
    sighost_counts = host_counts[host_counts>8]
    keepHosts = sighost_counts.index.map(lambda x: str(x))
    data = data[data["Host"].isin(keepHosts)]
    
    
    #Reduce the data - Remove IPs that are present less than 17 times
    IP_counts = data["IPadd"].value_counts()
    IP_counts
    IP_counts.describe()
    sigIP_counts = IP_counts[IP_counts>17]
    keepIPs = sigIP_counts.index.map(lambda x: str(x))
    data = data[data["IPadd"].isin(keepIPs)]
    return data

def get_host_counts(df,hostname,feat_name):
    '''
        Selects feature column out given hostname
    '''
    count = df[df["Host"]==hostname][feat_name].value_counts()
    return count 

def get_hosts_counts(df,hostnames,feat_name):
    '''
        Selects feature column out for list of hostnames
    '''
    count=data[data["Host"].isin(hostnames)][feat_name].value_counts()
    return count 

def naive_bayes_fraud(Fcounts,NFcounts,Tcounts):
    '''
        Runs naive bayes, or a modified version that returns a score, not a probability
    '''
    vocab = NFcounts.add(Fcounts,fill_value=0)
    Fprior = .625
    NFprior = .375
    prob_fraud = 0.0
    log_prob_fraud = 0.0
    prob_notfraud = 0.0
    log_prob_notfraud = 0.0    
    for ind, val in Tcounts.iteritems():
        if ind in vocab.index:
            p_value = (vocab[ind]+0.0)/vocab.sum()
            #print "Prob. of value: ", p_value
            #p_v_given_fraud = (Fcounts.get(ind,0.0)+0.0)/Fcounts.sum()
            p_v_given_fraud = (Fcounts.get(ind,0.0)+1.0)/(Fcounts.sum()+len(vocab))
            #print "Prob. of val | fraud: ", p_v_given_fraud
            #p_v_given_notfraud = (NFcounts.get(ind,0.0)+0.0)/NFcounts.sum()
            p_v_given_notfraud = (NFcounts.get(ind,0.0)+1.0)/(NFcounts.sum()+len(vocab))
            #print "Prob. of val | not fraud: ", p_v_given_notfraud
        else:
            p_value = 1.0/(vocab.sum()+1)
            p_v_given_fraud = 1.0/(Fcounts.sum()+len(vocab))
            p_v_given_notfraud = 1.0/(NFcounts.sum()+len(vocab))
        if p_v_given_fraud > 0:
            prob_fraud += (val * p_v_given_fraud) / p_value
            log_prob_fraud += math.log(val * p_v_given_fraud / p_value)
        if p_v_given_notfraud >0:
            prob_notfraud += (val * p_v_given_notfraud) / p_value
            log_prob_notfraud += math.log(val * p_v_given_notfraud / p_value)
    '''
    print "\nFraud Score:  ", (prob_fraud*Fprior)/(prob_fraud*Fprior+prob_notfraud*NFprior)
    print "SumProb. (fraud):  ", prob_fraud + Fprior
    print "SumProb. (not fraud):  ", prob_notfraud + NFprior
    print "LogScore (fraud):  ", log_prob_fraud + math.log(Fprior)
    print "LogScore (not fraud):  ", log_prob_notfraud + math.log(NFprior)
    print "Fscore : ", np.exp(log_prob_fraud+math.log(Fprior)-(log_prob_fraud+math.log(Fprior)+log_prob_notfraud+math.log(NFprior)))   
    exp_prob_fraud = np.exp(log_prob_fraud + math.log(Fprior))
    exp_prob_notfraud = np.exp(log_prob_notfraud + math.log(NFprior))
    print "Likelihood of Fraud(exp):  ", exp_prob_fraud/(exp_prob_fraud+exp_prob_notfraud)#(log_prob_fraud + math.log(Fprior))/(log_prob_fraud + math.log(Fprior)+log_prob_notfraud + math.log(NFprior))
    print "Likelihood of Fraud(log):  ", (log_prob_fraud + math.log(Fprior))/(log_prob_fraud + math.log(Fprior)+log_prob_notfraud + math.log(NFprior))
    '''
    return prob_fraud/(prob_fraud+prob_notfraud)

def run_fraudscore(df,host,feat_name):
    '''
        Runs naive_bayes_fraud for a host. Checks to see if it is in training set and removes it if True
    '''
    fraudtraffic=["featureplay.com","uvidi.com","spryliving.com","greatxboxgames.com",
              "mmabay.co.uk","workingmothertv.com","besthorrorgame.com","dailyparent.com","superior-movies.com",
              "yourhousedesign.com","outdoorlife.tv","drumclub.info","cycleworld.tv","hmnp.us","nlinevideos.com"]
    nfraudtraffic=["google.com","foxsports.com","washingtonpost.com","amazon.com",
               "nytimes.com","tvguide.com","pandora.com","youtube.com","cnn.com"]
    if host in fraudtraffic:
        fraudtraffic.remove(host)
    if host in nfraudtraffic:
        nfraudtraffic.remove(host)
    F_count = get_hosts_counts(data,fraudtraffic,feat_name)
    NF_count = get_hosts_counts(data,nfraudtraffic,feat_name)
    T_count = get_host_counts(data,host,feat_name)
    return naive_bayes_fraud(F_count,NF_count,T_count)

def clean_plot(color='w',ax=plt.gca(), leftAxisOn=True):
	# Make a cleaner, prettier plot
	ax.set_axis_bgcolor('w')
	ax.spines['bottom'].set_color(color)
	ax.spines['top'].set_color(color) 
	ax.spines['right'].set_color(color)
	ax.spines['left'].set_color(color)
	if leftAxisOn is True:
		ax.spines['left'].set_color((0.5,0.5,0.5))
		ax.yaxis.set_ticks_position('left')
		ax.get_yaxis().set_tick_params(direction='out',color=(0.5,0.5,0.5),length=3.5)


def runallhost_fraudscore(data,feat_name):
    # Run fraudscores for each host   
    allhosts = pd.Series(data["Host"].ravel()).unique()
    fraudscores = np.zeros_like(allhosts)
    for i,h in enumerate(allhosts):
        #print i, "out of", len(allhosts)
        fraudscores[i]=run_fraudscore(data,h,feat_name)

    host_score_prob = pd.DataFrame({'host' : allhosts,'fscore' : fraudscores})
    host_score_prob.sort('fscore',ascending=False)
    
    #sort the scores and reindex    
    sort_host_score_prob = host_score_prob.sort_index(by='fscore',ascending=False)
    sort_host_score_prob.index=range(1,len(sort_host_score_prob)+1)

    fraudtraffic=["featureplay.com","uvidi.com","spryliving.com","greatxboxgames.com",
              "mmabay.co.uk","workingmothertv.com","besthorrorgame.com","dailyparent.com","superior-movies.com",
              "yourhousedesign.com","outdoorlife.tv","drumclub.info","cycleworld.tv","hmnp.us","nlinevideos.com"]
       
    #plot results
    fig = plt.figure(figsize=(16,8))
    ax = plt.axes([.125, .2, .775, .7])
    plt.title('Fraud Scores from '+feat_name+' for known Fraudulent')
    plt.ylabel('Fraud Score Percentile', size=12)
    textFont  = {'family' : u'sans-serif',
         'size'   : 12,
         'style'  : u'italic' }		
    rects = plt.bar(np.arange(15),100-(sort_host_score_prob[sort_host_score_prob['host'].isin(fraudtraffic)].index/(len(sort_host_score_prob)+0.0))*100,color='r')
    plt.xticks(np.arange(15)+0.5, sort_host_score_prob[sort_host_score_prob['host'].isin(fraudtraffic)].host.values, rotation=60,
    	horizontalalignment='right', **textFont)
    
    for rect in rects:
        height = int(rect.get_height())
        rankStr = str(height)
        xloc = rect.get_x()+rect.get_width()/2.0
        yloc = 0.95*height
        ax.text(xloc, yloc, rankStr, horizontalalignment='center',
                verticalalignment='center', color='white', weight='bold')
    clean_plot(ax=ax)
    
    f = open('Integral_data_host_fraud_ranks'+feat_name+'_noreduction.txt', 'w')
    for ind, fsc, hst in sort_host_score_prob.itertuples():
        f.write(str(ind) + ' ' + hst + '\n')
    f.close()
    

if __name__ == "__main__":
    print 'Loading data'
    log_file = "D:\Downloads\Integral_data_set.tsv"
    data = pd.io.parsers.read_csv(log_file,sep='\t',names=['Timestamp','IPadd','Browser','UserA','Host','Iinview','Nplugins','Bwinpossize','NetLat'],header=None)
    
    print 'Cleaning data'
    data = clean_data(data)
    print 'Adding features'
    data = add_features(data)
    print 'Reducing data'
    data = reduce_data(data)
    
    for f in ['Browser','UserA','Iinview','Nplugins','Bwinarea']:
        print 'Computing fraud scores for '+f
        runallhost_fraudscore(data,f)
        plt.savefig('KnownFraudulent'+f+'FraudScore.pdf')
    



