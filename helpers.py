import polars as pl

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

