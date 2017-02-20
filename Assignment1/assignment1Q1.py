from yahoo_finance import Share
import numpy as np
import statsmodels.api as sm
import pandas as pd

'''this function convert the list of prices to the list of returns'''
def getLogReturn(prices):
    prev=None #set previous price
    returns=list() #store the list of return series
    for item in prices:
        if prev is None:
            prev=item #initialize previous price
        else:
            returns.append(np.log(prev/item)) #append the daily return to returns
            prev=item #update prev
    return returns

'''get the differences of two lists'''
def diffLists(list1,list2):
    len1=len(list1) #get length of list1
    len2=len(list2) #get length of list2
    if len1 != len2:  #if two lists do not have equal length, data has some error
        print("error")
        return
    else:
        return [list1[i]-list2[i] for i in range(len1)] #return the difference of lists

'''model time series using AR(p) model and write relevant parameters to files'''
def modelAR(filename,series,p_max): #write to filename and series is to be modelled in AR(p), p<=p_max
    plist=range(1,p_max+1) #store values of p
    aic=list() #store aic
    bic=list() #store bic
    minAIC=(None,None) #the first element of tuple is the corresponding p and the second is AIC value
    minBIC=(None,None) #the first element of tuple is the corresponding p and the second is BIC value
    for p in plist:
        arma=sm.tsa.ARMA(series,(p,0)).fit(method='mle',disp=False) #model time series
        #append to lists
        aic.append(arma.aic)
        bic.append(arma.bic)
        print(p)
        if minAIC[1] is None:
            minAIC=(p,arma.aic) #initialize minAIC
        else:
            if minAIC[1]>arma.aic:
                minAIC=(p,arma.aic) #update minAIC
                print(minAIC)
        if minBIC[1] is None:
            minBIC=(p,arma.bic) #initialize minBIC
        else:
            if minBIC[1]>arma.bic:
                minBIC=(p,arma.bic) #update minBIC
    the_dict={'AR(p)':plist,'AIC':aic,'BIC':bic} #construct a dictionary
    data=pd.DataFrame.from_dict(the_dict) #construct dataframe
    data=data[['AR(p)','AIC','BIC']] #rearrange columns
    data.to_csv(filename,index=None) #write to a csv file
    return (minAIC,minBIC)


def main():
    spy=Share('SPY') #get object about SPY
    iwv=Share('IWV') #get object about IWV
    end='2017-02-04' #set the end date
    start='2007-02-04' #set the start date
    spy_hist=spy.get_historical(start,end) #get historical data about spy
    iwv_hist=iwv.get_historical(start,end) #get historical data about iwv
    spy_adj=[float(dic['Adj_Close']) for dic in spy_hist] #store adjust close prices
    iwv_adj=[float(dic['Adj_Close']) for dic in iwv_hist] #store adjust close prices
    spy_return=getLogReturn(spy_adj)
    iwv_return=getLogReturn(iwv_adj)
    xt=diffLists(spy_return,iwv_return)
    result=modelAR("AR30.csv",xt,30)
    print(result)



if __name__=='__main__':
  main()
