import polars as pl

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_rolling_features(df):

    series_results = []
    series_ids = df["series_id"].unique().to_numpy()
    for series_id in series_ids:
        ds = df.filter(pl.col("series_id")==series_id).sort(by="dt_minute")
    
        rolling_features = []
        diff_rolling_features = []
        div_rolling_features = []
    
        vars = ['enmo','anglez']

        # rolling features for 15 min, 60 min, 3 hours and 8 hours
        times = [15, 60, 60*3, 60*8] 
        
        for mins in times:
            for var in vars:
                rolling_features += [
                    pl.col(var).rolling_mean(mins, center=True, min_periods=1).alias(f'{var}_{mins}m_mean'),
                    pl.col(var).rolling_max(mins, center=True, min_periods=1).alias(f'{var}_{mins}m_max'),
                    pl.col(var).rolling_min(mins, center=True, min_periods=1).alias(f'{var}_{mins}m_min'),
                    pl.col(var).rolling_std(mins, center=True, min_periods=1).alias(f'{var}_{mins}m_std'),
                    pl.col(var).diff().abs().rolling_std(mins, center=True, min_periods=1).alias(f'{var}_diffs_{mins}m_sum'),
        ]
                
        ds = ds.with_columns(
            rolling_features
        )
    
        for mins in times:
            for var in vars:
                div_rolling_features += [
                    (pl.col(f'{var}_diffs_{mins}m_sum') /
                      ((pl.col(f'{var}_{mins}m_max') - pl.col(f'{var}_{mins}m_min')).abs() +
                        pl.lit(1))).alias(f'{var}_diffs_sum_div_max_min_{mins}m'),
                ]
        
        ds = ds.sort(by="dt_minute")
        ds = ds.with_columns(
            div_rolling_features
        )
        series_results.append(ds)
    df = pl.concat(series_results).sort(by=["series_id","dt_minute"])
    return df


def concat_in_order(struct: dict) -> int:
    vars = [struct["index"], struct["variable"]]
    concat =  " ".join(sorted(vars))
    return concat


def create_sleeping_time_vars(df):
    df = df.with_columns(
        pl.when(pl.col("hour").is_in([22,23,0,1,2,3,4,5,6,7,8]))
        .then(1)
        .otherwise(0)
        .alias('night')
    )
    return df


def calculate_metrics(y_pred,y_val,target):
    precision = sum((y_pred==y_val) & (y_val==target)) / sum(y_pred==target)
    recall = sum((y_pred==y_val) & (y_val==target)) / sum(y_val==target)
    f1_score = 2 / ((1/precision) + (1/recall))
    return precision, recall, f1_score


def clean_predictions(df):

    df_pl = pl.from_pandas(df)

    series_reults = []
    
    for s in df_pl["series_id"].unique():
        df_s = df_pl.filter(pl.col('series_id')==s).sort(by='dt_minute')

         # Mark short breaks in sleep as sleep
        df_s = df_s.sort(by='dt_minute')
        df_s = df_s.sort(by=['series_id','dt_minute']).with_columns(
            pl.col('prediction').rolling_min(window_size=10, center=False).alias('long_sleep'),
            pl.col('step').rolling_min(window_size=10, center=False).alias('long_sleep_start'),
            pl.col('step').rolling_max(window_size=10, center=False).alias('long_sleep_end')
        )

        # Clean onsets - they all start 9 min late
        pred_onsets = df_s.filter((df_s['long_sleep'].diff()!=0)&(df_s['long_sleep']==1))['long_sleep_start'].to_list()
        for onset in pred_onsets:
            df_s = df_s.with_columns(
             pl.when((pl.col('step')<=onset+12*9)&(pl.col('step')>=onset)).then(1)
            .otherwise(pl.col('long_sleep'))
            .alias('long_sleep'))
            
        pred_onsets = df_s.filter((df_s['long_sleep'].diff()!=0)&(df_s['long_sleep']==1))['step'].to_list()
        pred_wakeups = df_s.filter((df_s['long_sleep'].diff()==-1)&(df_s['long_sleep']==0)|(df_s['long_sleep'].diff()==1)&(df_s['long_sleep']==2))['step'].to_list() 
        
        # mark breaks in sleep that are less or equal to 30 min as sleep
        short_break_periods = [(wakeup, onset) for onset, wakeup in zip(pred_onsets[1:], pred_wakeups[:-1]) if onset - wakeup <= 12 * 30]
        
        for break_start, break_end in short_break_periods:
            df_s = df_s.with_columns(
             pl.when((pl.col('step')>=break_start)&(pl.col('step')<=break_end)).then(1)
            .otherwise(pl.col('long_sleep'))
            .alias('long_sleep'))

        # Ignore signle short sleep times
        pred_onsets = df_s.filter((df_s['long_sleep'].diff()!=0)&(df_s['long_sleep']==1))['long_sleep_start'].to_list()
        pred_wakeups = df_s.filter((df_s['long_sleep'].diff()==-1)&(df_s['long_sleep']==0)|(df_s['long_sleep'].diff()==1)&(df_s['long_sleep']==2))['long_sleep_end'].to_list() 
        short_sleep_times = [(onset, wakeup) for onset, wakeup in zip(pred_onsets, pred_wakeups) if wakeup - onset <= 12 * 30]
        for onset, wakeup in short_sleep_times:
            df_s = df_s.with_columns(
             pl.when((pl.col('step')>=onset)&(pl.col('step')<=wakeup)).then(0)
            .otherwise(pl.col('long_sleep'))
            .alias('long_sleep'))
        
        series_reults.append(df_s)
    
    df = pl.concat(series_reults).to_pandas()
    return df


def get_events(df, idx=None, target='long_sleep'):
    if not idx:
        idx = df['series_id'].unique()[0]
        
    df = df.filter(pl.col("series_id")==idx)
    events = pl.DataFrame(schema={'night_count':int, 'series_id':str, 'step':int, 'timestamp': pl.Datetime, 'event':str, 'score':float})
    
    pred_onsets = df.filter((df[target].diff()!=0)&(df[target]==1))['step'].to_list()
    pred_wakeups = df.filter((df[target].diff()==-1)&(df[target]==0)|(df[target].diff()==1)&(df[target]==2))['step'].to_list()
    sleep_periods = [(onset, wakeup) for onset, wakeup in zip(pred_onsets, pred_wakeups)]

    night_count = 0
    for onset, wakeup in sleep_periods:
        night_count += 1
        # Scoring using mean probability over period
        score = df.filter((pl.col('step') >= onset) & (pl.col('step') <= wakeup))['probability'].mean()
        onset_time = df.filter(pl.col("step")==onset)["dt_minute"][0]
        wakeup_time = df.filter(pl.col("step")==wakeup)["dt_minute"][0]
        # Adding sleep event to dataframe
        events = events.vstack(pl.DataFrame().with_columns(
            pl.Series([night_count, night_count]).alias('night_count'), 
            pl.Series([idx, idx]).alias('series_id'), 
            pl.Series([onset, wakeup]).alias('step'),
            pl.Series([onset_time, wakeup_time]).alias('timestamp'),
            pl.Series(['onset', 'wakeup']).alias('event'),
            pl.Series([score, score]).alias('score')
        ))
    return events


def plot_data(df, idx, labels="target"):

    print(idx)
    df = df.to_pandas()
    df = df.loc[df['series_id']==idx].sort_values(by='dt_minute')

    fig = make_subplots(specs=[[{'secondary_y': True}]])
    
    # Plotting time series data
    fig.add_trace(go.Scatter(x=df['dt_minute'], y=df['anglez'], mode='lines', name='anglez'), secondary_y=False)
    fig.add_trace(go.Scatter(x=df['dt_minute'], y=df['enmo'], mode='lines', name='enmo'), secondary_y=True)
    
    # Plotting sleeping periods
    onsets = df.loc[(df[labels].diff()!=0)&(df[labels]==1)]['dt_minute'].to_list()
    wakeups = df.loc[(df[labels].diff()==-1)&(df[labels]==0)|(df[labels].diff()==1)&(df[labels]==2)]['dt_minute'].to_list()
    for onset, wakeup in zip(onsets, wakeups) :
        fig.add_vrect(
            x0=onset, x1=wakeup,
            fillcolor='gray', opacity=0.2,
            layer='below', line_width=0
        )
        
    # Add dummy trace for the legend
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='lines',
        line=dict(color='gray', width=0),
        fill='tozeroy',
        fillcolor='gray',
        opacity=0.2,
        name="Sleeping periods"
        ))
    
    # Plotting notwear periods
    not_wear_starts = df.loc[(df[labels].diff()>0)&(df[labels]==2)]['dt_minute'].to_list()
    not_wear_ends = df.loc[(df[labels].diff()==-2)&(df[labels]==0)|(df[labels].diff()==-1)&(df[labels]==1)]['dt_minute'].to_list()

    # if all time not-wear
    stats = df.groupby("target").dt_minute.count()
    not_wear_num = stats.loc[stats.index==2]
    if not not_wear_num.empty and df.shape[0] == not_wear_num.tolist()[0]:
        not_wear_starts = [df["dt_minute"].min()]
        not_wear_ends = [df["dt_minute"].max()]
        
    if len(not_wear_ends) < len(not_wear_starts):
        not_wear_ends += [df["dt_minute"].max()]
    for start, end in zip(not_wear_starts, not_wear_ends):
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor='violet', opacity=0.2,
            layer='below', line_width=0
        )
        
    # Add dummy trace for the legend
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='lines',
        line=dict(color='violet', width=0),
        fill='tozeroy',
        fillcolor='violet',
        opacity=0.7,
        name="Not-wear periods"
        ))

    fig.update_layout(
        title_text=f"Accelerometer Data - Full Minutes Averages<br><sub>series_id: {idx}</sub>", 
        #template = 'ggplot2',
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), yaxis2=dict(showgrid=False)
        )
    
    fig.show()
