from datetime import timedelta, datetime
from matplotlib.pyplot import xlabel
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

pd.options.display.float_format = "{:,}".format

st.set_page_config(layout="wide")

st.write('# Stock Network Analysis')

with st.expander(label="Project Descriptions", expanded=False):
    st.markdown(
        body="""
    #### TOC
    1. 
    2. 
    3. 
    4. 
    5.
    6.
    
    """
    )

# Using object notation
# add_selectbox = st.sidebar.selectbox(
#     "How would you like to be contacted?", ("Email", "Home phone", "Mobile phone")
# )
# st.button("Click me")
# st.checkbox("I agree")
# st.radio("Pick one", ["cats", "dogs"])
# st.selectbox("Pick one", ["cats", "dogs"])
# st.multiselect("Buy", ["milk", "apples", "potatoes"])
# st.slider("Pick a number", 0, 100)
# st.select_slider("Pick a size", ["S", "M", "L"])
# st.text_input("First name")

# st.time_input("Meeting time")
# st.download_button("Download file", data)

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

#st.dataframe(asset_returns)

# 조회 기간은 전역적으로 변경 가능하게 제공
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

st.dataframe(sector_returns)

#creates a list of etfs
etfs = sector_returns.columns.tolist()

# Moving Window
def moving_window(df, length):
    return [df[i:i+length] for i in range(0, (len(df)+1)-length, length)]

sector_mw = moving_window(sector_returns, 21)
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

dates = [list(sector_mw[i].index)[0].strftime('%Y-%m-%d') for i in range(len(sector_mw))]

dict_x, dict_y, dict_size = {}, {}, {}
for etf in etfs : 
  dict_x[etf] = []
  dict_y[etf] = []
  dict_size[etf] = []

for H in mw_net :
#   edges, weights = zip(*nx.get_edge_attributes(H, "weight").items())  
  pos = nx.kamada_kawai_layout(H)
  for k, v in pos.items():
    dict_x[k].append(v[0])
    dict_y[k].append(v[1])
  deg = H.degree
  for n, d in deg:
    dict_size[n].append(d)


df_x = pd.DataFrame(dict_x)
df_y = pd.DataFrame(dict_y)
df_size = pd.DataFrame(dict_size)

def get_edge(idx):
    H = mw_net[idx]
    edge_x = []
    edge_y = []
    edges, weights = zip(*nx.get_edge_attributes(H, "weight").items())
    for node0, node1 in edges:
        x0, x1 = df_x.loc[idx, node0], df_x.loc[idx, node1]
        y0, y1 = df_y.loc[idx, node0], df_y.loc[idx, node1]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
    return edge_x, edge_y, weights

def make_edge(x, y, width): # text, 
    return  go.Scatter(x         = x,
                       y         = y,
                       line      = dict(width = width,
                                   color = 'gray'), # 'cornflowerblue'
                    #    hoverinfo = str(width),
                    #    text      = ([str(width)]),
                       mode      = 'lines')

# make figure
fig_dict = {
    "data": [],
    "layout": {},
    "frames": []
}

# fill in most of layout
fig_dict["layout"]["width"] = 600
fig_dict["layout"]["height"] = 600

fig_dict["layout"]["xaxis"] = {"range": [-1.1, 1.1], "title": ""} # , "range": [35, 80]
fig_dict["layout"]["yaxis"] = {"range": [-1.1, 1.1], "title": "" } #, "type": "log"

fig_dict["layout"]["hovermode"] = "closest"
fig_dict["layout"]["updatemenus"] = [
    {
        "buttons": [
            {
                "args": [None, {"frame": {"duration": 1000, "redraw": False},
                                "fromcurrent": True, "transition": {"duration": 1000,
                                                                    "easing": "quadratic-in-out"}}],
                "label": "Play",
                "method": "animate"
            },
            {
                "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                  "mode": "immediate",
                                  "transition": {"duration": 0}}],
                "label": "Pause",
                "method": "animate"
            }
        ],
        "direction": "left",
        "pad": {"r": 10, "t": 87},
        "showactive": False,
        "type": "buttons",
        "x": 0.1,
        "xanchor": "right",
        "y": 0,
        "yanchor": "top"
    }
]

sliders_dict = {
    "active": 0,
    "yanchor": "top",
    "xanchor": "left",
    "currentvalue": {
        "font": {"size": 20},
        "prefix": "Date:",
        "visible": True,
        "xanchor": "center" # right
    },
    "transition": {"duration": 300, "easing": "cubic-in-out"},
    "pad": {"b": 10, "t": 87},
    "len": 1.0, # 0.9
    "x": 0.1,
    "y": 0,
    "steps": []
}

# make data
edge_x, edge_y, weights = get_edge(0)
edge_dict = {
    "x": edge_x,
    "y": edge_y,
    "mode": "lines",
    "text": [],
    "line": {
        "color": "gray", "width":0.5
    },
    "name" : "edge"
}        
fig_dict["data"].append(edge_dict)

for etf in etfs :

    node_dict = {
        "x": [df_x[etf][0]],
        "y": [df_y[etf][0]],
        "mode": "markers",
        "text": [etf],
        "marker": {
            "sizemode": "area",
            "sizeref": 200000,
            "size": [df_size[etf][0]**8]
        },
        "name": etf
    }
    fig_dict["data"].append(node_dict)

# make frames
for idx, date in enumerate(dates):
    frame = {"data": [], "name": str(date)}
    
    edge_x, edge_y, weights = get_edge(idx)
    edge_dict = {
        "x": edge_x,
        "y": edge_y,
        "mode": "lines",
        "text": [],
        "line": {
            "color": "gray", "width":0.5
        },
        
    }        
    frame["data"].append(edge_dict)
    
    for etf in etfs :

        node_dict = {
            "x": [df_x[etf][idx]],
            "y": [df_y[etf][idx]],
            "mode": "markers",
            "text": [etf],
            "marker": {
                "sizemode": "area",
                "sizeref": 200000,
                "size": [df_size[etf][idx]**8]
            },
            "name": etf
        }

        frame["data"].append(node_dict)

    fig_dict["frames"].append(frame)
    slider_step = {"args": [
        [date],
        {"frame": {"duration": 300, "redraw": False},
         "mode": "immediate",
         "transition": {"duration": 300}}
    ],
        "label": date,
        "method": "animate"}
    sliders_dict["steps"].append(slider_step)


fig_dict["layout"]["sliders"] = [sliders_dict]

fig = go.Figure(fig_dict)

st.markdown('## Plotly Chart')
st.plotly_chart(fig)

from pyvis.network import Network    

nt = Network(height="750px", width="100%")
nt.from_nx(mw_net[0])
nt.show('./html/nx0.html')

nt = Network(height="750px", width="100%")
nt.from_nx(mw_net[1])
nt.show('./html/nx1.html')



import streamlit.components.v1 as components
st.sidebar.title('choose what time you want to see')
option = st.sidebar.selectbox('select graph', ('nx0', 'nx1'))

if option == 'nx0':
    HtmlFile = open("./html/nx0.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read()
    st.markdown('## Pyvis chart of nx0') 
    components.html(source_code, height = 1200,width=1000)    

if option == 'nx1':
    HtmlFile = open("./html/nx1.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    st.markdown('## Pyvis chart of nx1')
    components.html(source_code, height = 1200,width=1000)