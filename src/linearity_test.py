import pandas as pd
import matplotlib.pyplot as plt

#read in data to create cuts and visualze linearity
repeat_purch = pd.read_csv('/Users/etownbetty/Documents/Galvanize/Project/data/purchaseDataAnalysisSet.csv')
#cut the continuous data into bins to see if there is a general trend with respect to the outcome
repeat_purch['FirstOrderDiscr'] = pd.cut(repeat_purch['FirstOrderTotal'], 10, labels=range(1,11))
repeat_purch['FirstItemsDiscr'] = pd.cut(repeat_purch['FirstItemCnt'], 10, labels=range(1,11))

FirstOrderTotalBins = repeat_purch.groupby('FirstOrderDiscr')['FirstOrderTotal', 'repeat'].mean()
FirstOrderItemBins = repeat_purch.groupby('FirstItemsDiscr')['FirstItemCnt', 'repeat'].mean()

#plot the average for the cuts and outcome
plt.scatter(FirstOrderTotalBins.FirstOrderTotal, FirstOrderTotalBins.repeat)
plt.xlabel('First Purchase Spend')
plt.ylabel('Probability of Repeat Customer')
plt.title('Probability of Repeat Customer VS First Purchase Spend')
plt.savefig('FirstOrderTotalVsProbRepeatCustomer.png')
plt.close()

plt.scatter(FirstOrderItemBins['FirstItemCnt'], FirstOrderItemBins['repeat'])
plt.xlabel('First Purchase Item Count')
plt.ylabel('Probability of Repeat Customer')
plt.title('Probability of Repeat Customer VS First Item Cnt')
plt.savefig('FirstOrderItemCntVsProbRepeatCustomer.png')
plt.close()
