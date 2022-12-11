from datetime import timedelta, datetime
from matplotlib.pyplot import xlabel
from plotly.subplots import make_subplots

import plotly.express as px
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

pd.options.display.float_format = "{:,}".format
st.set_page_config(layout="wide")
st.markdown('# Stock Network Analysis')

with st.expander(label="Project Descriptions", expanded=False):
    st.markdown(
        body="""
    #### TOC
    1. Data
    2. Correlation Matrix
    3. Correlation Networks
    4. Correlation Similarity
    5. Interactive Correlation Graph
    6. Market Analysis
    """
    )

sector_etfs = {
    'XLE US Equity': 'Energy',
    'XLU US Equity': 'Utilities', 
    'XLK US Equity': 'Information Technology', 
    'XLB US Equity': 'Materials',
    'XLP US Equity': 'Consumer Staples', 
    'XLY US Equity': 'Consumer Discretionary',
    'XLI US Equity': 'Industrials',
    'XLV US Equity': 'Health Care',
    'XLF US Equity': 'Financials',
    'VOX US Equity': 'Communication Services', # 'XLC US Equity': 'Communication Services',
    #'XLRE US Equity': 'Real Estate'
}

benchmark = 'SPX Index' # S&P500

asset_returns = pd.read_excel('BLOOMBERG_ASSET_TR_RETURNS.xlsx', index_col=0)

# 조회 기간은 전역적으로 변경 가능하게 제공
st.markdown('### Select Date Range')
drange = st.date_input(
    "조회할 기간 지정",
    min_value=asset_returns.index.min(),
    max_value=asset_returns.index.max(),
    value=[datetime(2006,12,31), asset_returns.index.max()],
)

etf_returns = asset_returns[list(sector_etfs.keys())]
first_date = datetime(2006, 12, 31)
sector_returns = etf_returns.loc[first_date:, etf_returns.columns != benchmark].dropna(how='all').fillna(0)
bm_returns = asset_returns.loc[first_date:, benchmark].dropna(how='all').fillna(0)

#creates a list of etfs
etfs = sector_returns.columns.tolist()

#-----------------------------------------------------
# Moving Window
def moving_window(df, length, overlaps=10):
    return [df[i:i+length] for i in range(0, (len(df)+1)-length, length//overlaps)]

sector_mw = moving_window(sector_returns, 21)

#-----------------------------------------------------
# Distance Correlation
import dcor
def df_distance_correlation(df_list):
    
    df_corr_list = []
    for x in range(0, len(df_list)):
        #initializes an empty DataFrame
        df_dcor = pd.DataFrame(index=etfs, columns=etfs)

        #initialzes a counter at zero
        k=0

        #iterates over the time series of each stock
        for i in etfs:

            #stores the ith time series as a vector
            v_i = df_list[x].loc[:, i].values

            #iterates over the time series of each stock subect to the counter k
            for j in etfs[k:]:

                #stores the jth time series as a vector
                v_j = df_list[x].loc[:, j].values

                #computes the dcor coefficient between the ith and jth vectors
                dcor_val = dcor.distance_correlation(v_i, v_j)

                #appends the dcor value at every ij entry of the empty DataFrame
                df_dcor.at[i,j] = dcor_val

                #appends the dcor value at every ji entry of the empty DataFrame
                df_dcor.at[j,i] = dcor_val

            #increments counter by 1
            k+=1
        df_corr_list.append(df_dcor)
    #returns a DataFrame of dcor values for every pair of stocks
    return df_corr_list

mw_corr = df_distance_correlation(sector_mw)


#-----------------------------------------------------
# Networkx Transformation
import networkx as nx
def build_corr_nx(df_corr_list):
    
    net_list = []
    for x in range(0, len(df_corr_list)):
        # converts the distance correlation dataframe to a numpy matrix with dtype float
        cor_matrix = df_corr_list[x].values.astype('float')

        # Since dcor ranges between 0 and 1, (0 corresponding to independence and 1
        # corresponding to dependence), 1 - cor_matrix results in values closer to 0
        # indicating a higher degree of dependence where values close to 1 indicate a lower degree of 
        # dependence. This will result in a network with nodes in close proximity reflecting the similarity
        # of their respective time-series and vice versa.
        sim_matrix = 1 - cor_matrix

        # transforms the similarity matrix into a graph
        G = nx.from_numpy_matrix(sim_matrix)

        # extracts the indices (i.e., the stock names from the dataframe)
        stock_names = df_corr_list[x].index.values

        # relabels the nodes of the network with the stock names
        G = nx.relabel_nodes(G, lambda x: stock_names[x])

        # assigns the edges of the network weights (i.e., the dcor values)
        G.edges(data=True)

        # copies G
        ## we need this to delete edges or othwerwise modify G
        H = G.copy()

        # iterates over the edges of H (the u-v pairs) and the weights (wt)
        for (u, v, wt) in G.edges.data('weight'):
            # selects edges with dcor values less than or equal to 0.33
            if wt >= 1 - 0.325:
                # removes the edges 
                H.remove_edge(u, v)

            # selects self-edges
            if u == v:
                # removes the self-edges
                H.remove_edge(u, v)
        
        net_list.append(H)
    # returns the final stock correlation network            
    return net_list

mw_net = build_corr_nx(mw_corr)

#-----------------------------------------------------

def calculate_metric(df_corr_list, type = 'sum'):
    
    metric_list = []

    for x in range(0, len(df_corr_list)): #len(df_corr_list)

        cor_matrix = df_corr_list[x].values.astype('float') # np matrix
        sim_matrix = 1 - cor_matrix # similarity matrix (distance matrix), diagonal is 0

        if(type == 'sum'):
            metric = np.sum(sim_matrix)
        elif(type == 'max-min'):
            max = np.max(sim_matrix)
            min = np.min(sim_matrix + np.eye(sim_matrix.shape[0]))
            metric = max - min

        elif(type == 'entropy'):
            # Normalize the matrix into probability distribution, then get rid of diagonal (all 0).
            # Then Calculate entropy with the probability distribution.
            
            sim_matrix /= np.sum(sim_matrix)
            metric = 0

            for i in range(sim_matrix.shape[0]):
                for j in range(sim_matrix.shape[1]):
                    if i != j:
                        metric += -(sim_matrix[i][j] * np.log(sim_matrix[i][j]))

        metric_list.append(metric)
       
    return metric_list

def calculate_diff(metric_list, h = 12, overlaps = 6):

    diff = []
    for i in range(0,len(metric_list) - h,h//overlaps):
        diff.append(metric_list[i+h] - metric_list[i])

    return diff

#-----------------------------------------------------

st.write('Project Overview Flow Fig')
data1, data2, data3 = st.columns(3)
with data1:
    st.markdown('### Preprocessed Data')
    st.dataframe(sector_returns)
with data2:
    st.markdown('### Correlation Matrix')
    st.dataframe(mw_corr[0])
with data3:
    st.markdown('### Correlation Network')
    st.write(mw_net[0])

#-----------------------------------------------------
st.markdown('## Dashboard Flow')

st.markdown('### Select Date')
dates = [list(sector_mw[i].index)[0].strftime('%Y-%m-%d') for i in range(len(sector_mw))]
import datetime as dt
dates_list = [dt.datetime.strptime(date, '%Y-%m-%d').date() for date in dates]

# Create the slidebar with range in REF
selected_date = st.select_slider(
    label="Selected date : ",
    options=dates_list
)
st.markdown('## Market Chart')
# Line Chart
avg_sector_returns = pd.DataFrame(columns = sector_returns.columns)
for sector in sector_mw :
    avg_sector_returns.loc[sector.index[0]] = sector.sum()

bm_mw = moving_window(bm_returns, 21)
avg_bm = pd.DataFrame(columns=[benchmark])
for bm in bm_mw :
    avg_bm.loc[bm.index[0]] = bm.sum()

avg_bm = avg_bm.reset_index()
avg_bm.columns = ['Dates', benchmark]

chart1, chart2 = st.columns(2)
with chart1:
    st.markdown("### Market(S&P) Line Chart")
    fig = go.Figure(layout=go.Layout(width=900))
    fig.add_trace(go.Scatter(x=avg_bm['Dates'], y=avg_bm[benchmark], name=benchmark))
    fig.add_vline(x=selected_date, line_width=3, line_dash="dash")
    st.plotly_chart(fig)
with chart2:
    st.markdown('### Sector Line Chart')
    default_etfs = ['XLE US Equity', 'XLF US Equity']
    select_etfs = st.multiselect("Select ETF sectors you want to display", etfs, default_etfs) 
    dfs = {select_etf : avg_sector_returns[select_etf] for select_etf in select_etfs}

    fig = go.Figure(layout=go.Layout(width=900))
    for default_etf in default_etfs:
        fig = fig.add_trace(go.Scatter(x=avg_sector_returns.index, y=avg_sector_returns[default_etf], name=default_etf))


    fig = go.Figure(layout=go.Layout(width=900))
    for select_etf, df in dfs.items():
        fig = fig.add_trace(go.Scatter(x=avg_sector_returns.index, y=avg_sector_returns[select_etf], name=select_etf))
    fig.add_vline(x=selected_date, line_width=3, line_dash="dash")
    fig.update_xaxes(title_text = "Dates")
    fig.update_yaxes(title_text= "Rate of Return")
    st.plotly_chart(fig)
#-----------------------------------------------------
# Correlation Graph Analysis
st.markdown('## Correlation Chart Analysis')
## 재훈님 이 부분 채워주세요!

sector_mw = moving_window(sector_returns, 60, overlaps = 30)

mw_corr = df_distance_correlation(sector_mw)
mw_net = build_corr_nx(mw_corr)

metric = {}
metric['sum'] = calculate_metric(mw_corr, 'sum')
metric['entropy'] = calculate_metric(mw_corr, 'entropy')
metric['max-min'] = calculate_metric(mw_corr, 'max-min')
metric['diff'] = calculate_diff(metric['max-min'])

# max-min 이 최대, 최소를 찍는 date 를 저장하고 있는 부분
# 이 두 시점의 그래프를 그려보면 좋을 것 같아요.
min_ind = np.argmin(metric['max-min'])
max_ind = np.argmax(metric['max-min'])

fig = go.Figure(layout=go.Layout(width=1500))

fig.add_trace(go.Scatter(x=avg_sector_returns.index, y= metric['sum']/np.max(metric['sum']), name='sum'))
# fig.add_trace(go.Scatter(x=avg_sector_returns.index, y= (metric['entropy'] - np.average(metric['entropy'])) * 5, name='entropy'))
fig.add_trace(go.Scatter(x=avg_sector_returns.index, y= metric['max-min']/np.max(metric['max-min']), name='max-min'))
fig.add_trace(go.Scatter(x=avg_sector_returns.index[:-12:2], y= metric['diff']/np.max(metric['diff'])/3, name='diff'))

fig.update_xaxes(title_text = "Dates")
fig.update_yaxes(title_text= "Metrics")
st.plotly_chart(fig)

#-----------------------------------------------------
from pyvis.network import Network


#for i in range(len(mw_net)):
#    nt = Network(height="500px", width="100%")
#    nt.from_nx(mw_net[i])
#    nt.show(f'./html/{dates_list[i]}.html')


import streamlit.components.v1 as components
st.markdown(f'## Search Similar Market Trend Histories with Graph')
import numpy as np

def weighted_jaccard(g1, graph_list):
    # graph_list : graph of all windows
    # g1 : a single graph to be compared
    # g2 : a single graph to be conpared with g1
    sims = []
    for g2 in graph_list:
        edges = set(g1.edges()).union(g2.edges())
        mins, maxs = 0, 0
        for edge in edges:
            weight1 = g1.get_edge_data(*edge, {}).get('weight', 0)
            weight2 = g2.get_edge_data(*edge, {}).get('weight', 0)

            mins += min(weight1, weight2)
            maxs += max(weight1, weight2)
        sims.append(mins / (maxs+1.e-5))
    return sims

top3_sim  = []                                              # 날짜당 비슷한 graph의 인덱스를 3개씩
for single_graph in mw_net:
    total_sim = weighted_jaccard(single_graph, mw_net)
    top3_sim .append(np.argsort(total_sim)[::-1][1:4])

top3_dates = {key_date : list(np.array(dates)[single_top3_sim]) for (key_date, single_top3_sim) in zip(dates, top3_sim)}


tab1, tab2 = st.columns(2)
with tab1:
    st.markdown(f'### Pyvis Graph of {selected_date}') 
    HtmlFile = open(f"./html/{selected_date}.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read()
    components.html(source_code, height = 600,width=600)
with tab2:
    st.markdown(f'### Similar markets with {selected_date}')
    st.markdown(f'#### 1st candidate : {top3_dates[str(selected_date)][0]}')
    HtmlFile = open(f"./html/{top3_dates[str(selected_date)][0]}.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read()
    components.html(source_code, height = 360,width=360)
    st.markdown(f'#### 2nd candidate : {top3_dates[str(selected_date)][1]}')
    HtmlFile = open(f"./html/{top3_dates[str(selected_date)][1]}.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read()
    components.html(source_code, height = 360,width=360)
    st.markdown(f'#### 3rd candidate : {top3_dates[str(selected_date)][2]}')
    HtmlFile = open(f"./html/{top3_dates[str(selected_date)][2]}.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read()
    components.html(source_code, height = 360,width=360)