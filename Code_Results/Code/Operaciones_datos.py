import pandas as pd

def frecuencia_usuario(df, time, frecuencia, format_date):
    if frecuencia == 'M':
        frecuencia2 = 'MS'
    else :
        frecuencia2=frecuencia
    df[time] = pd.to_datetime(df[time], format=format_date)
    df = df.set_index(time)
    df = df.sort_index()
    df = df.asfreq(frecuencia2)
    df.index = df.index.to_period(frecuencia)
    return df