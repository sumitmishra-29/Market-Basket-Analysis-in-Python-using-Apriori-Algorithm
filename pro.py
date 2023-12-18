import pandas as pd
import numpy as np
from apyori import apriori
df = pd.read_csv('data.csv',header=None)
df.head()
df.fillna(0,inplace=True)
transactions = []

for i in range(0,len(df)):
    transactions.append([str(df.values[i,j]) for j in range(0,20) if str(df.values[i,j])!='0'])
transactions[0]
transactions[1]
rules = apriori(transactions, min_support=0.003, min_confidance=0.2, min_lift=3, min_length=2)
rules
Results = list(rules)
Results
df_results = pd.DataFrame(Results)
df_results.head()
support = df_results.support
first_values = []
second_values = []
third_values = []
fourth_value = []

# loop number of rows time and append 1 by 1 value in a separate list.. 
# first and second element was frozenset which need to be converted in list..
for i in range(df_results.shape[0]):
    single_list = df_results['ordered_statistics'][i][0]
    first_values.append(list(single_list[0]))
    second_values.append(list(single_list[1]))
    third_values.append(single_list[2])
    fourth_value.append(single_list[3])
lhs = pd.DataFrame(first_values)
rhs = pd.DataFrame(second_values)

confidance=pd.DataFrame(third_values,columns=['Confidance'])

lift=pd.DataFrame(fourth_value,columns=['lift'])
df_final = pd.concat([lhs,rhs,support,confidance,lift], axis=1)
df_final
df_final.fillna(value=' ', inplace=True)
df_final.head()
df_final.columns = ['lhs',1,'rhs',2,3,'support','confidance','lift']
df_final.head()
df_final['lhs'] = df_final['lhs'] + str(", ") + df_final[1]

df_final['rhs'] = df_final['rhs']+str(", ")+df_final[2] + str(", ") + df_final[3]
df_final.head()
df_final.drop(columns=[1,2,3],inplace=True)
df_final.head()
df_final.sort_values('lift', ascending=False).head(10)
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

df1 = pd.read_csv('data-2.csv', encoding="ISO-8859-1")
df1.head()
df1.Country.value_counts().head(5)
df1 = df1[df1.Country == 'France']
df1['Description'] = df1['Description'].str.strip()
df1 = df1[df1.Quantity >0]
df1.head(10)
basket = pd.pivot_table(data=df1,index='InvoiceNo',columns='Description',values='Quantity', aggfunc='sum',fill_value=0)
basket.head()
basket['10 COLOUR SPACEBOY PEN'].head(10)
def convert_into_binary(x):
    if x > 0:
        return 1
    else:
        return 0
basket_sets = basket.applymap(convert_into_binary)
basket_sets['10 COLOUR SPACEBOY PEN'].head(10)
print(basket_sets['POSTAGE'].head())

basket_sets.drop(columns=['POSTAGE'],inplace=True)
frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)
frequent_itemsets
rules_mlxtend = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules_mlxtend.head()
rules_mlxtend[ (rules_mlxtend['lift'] >= 4) & (rules_mlxtend['confidence'] >= 0.8) ].head()
