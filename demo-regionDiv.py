import streamlit as st
import chardet
import os
import time
import base64
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from functools import wraps
from shapely.geometry import Point

lonlat_str_format = {'经度': 'string', '纬度': 'string'}
lonlat_float_format = {'经度': 'float64', '纬度': 'float64'}


def time_costing(step):
    def func_name(func):
        @wraps(func)
        def core(*args, **kwargs):
            start = time.time()
            res = func(*args, **kwargs)
            region_name = args[2]
            if isinstance(region_name, str):
                region_name = [region_name]
            elif isinstance(region_name, list):
                pass
            st.write('、'.join(region_name) + '已划分')
            st.write(f'{step}耗时：{float(time.time() - start):.3f}秒')
            return res
        return core
    return func_name


def clean_lotlan(df_cell):
    for col_name in list(df_cell.loc[:, ['经度', '纬度']]):
        df_cell[col_name] = df_cell.astype({col_name: 'string'})[col_name].str.replace(r'\s', '', regex=True)
    df_cell_split_list = df_cell['经度'].str.contains('/')
    df_cell_split = df_cell[df_cell_split_list]
    if not df_cell_split.empty:
        df_comb = pd.DataFrame([], index=df_cell_split.index)
        for col_name in list(df_cell_split.loc[:, ['经度', '纬度']]):
            df_comb = pd.concat([df_comb, df_cell_split[col_name].str.split('/', expand=True).stack().reset_index(level=1).rename(columns={0: col_name})], axis=1)
        df_cell = pd.concat([df_cell[~df_cell_split_list], df_cell_split.iloc[:, :3].join(df_comb.drop(['level_1'], axis=1))]).reset_index(drop=True)
    return df_cell


def pd_read(file, extension, encode_n=None):
    try:
        if extension == 'csv':
            return pd.read_csv(file, dtype=lonlat_str_format, encoding=encode_n, low_memory=False)
        elif extension in ['xlsx', 'xls']:
            return pd.read_excel(file, dtype=lonlat_str_format)
        else:
            raise ValueError('文件格式错误')
    except ValueError:
        raise ValueError('文件读取错误')


@st.cache
def read_df(file):
    f_ext = file.name.split('.')[1]
    if f_ext == 'csv':
        encode = str.lower(chardet.detect(file.readline())["encoding"]).replace('-', '_')
        file.seek(0)
        if encode == 'utf-8':
            df = pd_read(file, f_ext, 'utf-8')
        elif encode == 'gb2312':
            try:
                df = pd_read(file, f_ext, 'gbk')
            except UnicodeDecodeError:
                df = pd_read(file, f_ext, 'gb18030')
        elif encode == 'utf_8_sig':
            df = pd_read(file, f_ext, 'utf_8_sig')
        elif encode == "iso-8859-1":
            df = pd_read(file, f_ext, 'gbk')
        else:
            raise ValueError('文件编码错误')
    elif f_ext in ['xlsx', 'xls']:
        df = pd_read(file, f_ext)
    else:
        raise ValueError('文件格式错误')

    # print(df.head(20))
    if df is not None and {'经度', '纬度'}.issubset(set(list(df))):
        df = df.pipe(clean_lotlan).astype(lonlat_float_format)
    else:
        raise ValueError('当前表格格式错误')
    return df


def layer_check(geofolder, key_list=None):
    try:
        dictionary = dict(pd.read_csv(f'.//{geofolder}//图层信息.csv', encoding='gb18030').loc[:, ['字段名称', '图层名称']].values)
        if (key_list is None) or (~set(key_list).issubset(dictionary.keys())):
            key_list = dictionary.keys()
        else:
            pass
        file_extension = 'shp' if geofolder == 'mapinfo' else 'gpkg'
        for index, name in enumerate(key_list):
            col_nm = list(gpd.read_file(f'.//{geofolder}//{dictionary[name]}.{file_extension}', encoding='utf-8'))
            if name not in col_nm:
                raise ValueError(f'图层字段<{name}>不在图层<{dictionary[name]}.{file_extension}>中')
        return dictionary
    except IOError:
        raise ValueError(f'找不到图层信息')


@time_costing('区域划分')
def region_division(df, region_dictionary, region_name, geofolder='qgis', lanlot_cols=('经度', '纬度')):
    lanlot_cols = list(lanlot_cols)
    if isinstance(region_name, str):
        region_name = [region_name]
    elif isinstance(region_name, list):
        pass
    else:
        raise ValueError('错误：区域名称错误')
    file_extension = 'shp' if geofolder == 'mapinfo' else 'gpkg'
    df_dropdu = df.drop_duplicates(subset=lanlot_cols).reset_index(drop=True)
    for index, name in enumerate(region_name):
        gdf_region = gpd.read_file(f'.//{geofolder}//{region_dictionary[name]}.{file_extension}', encoding='utf-8')
        gdf_region = gdf_region.to_crs('EPSG:2381') if gdf_region.crs is None else gdf_region.to_crs('EPSG:2381')
        lanlot = gpd.GeoSeries([Point(x, y) for x, y in zip(df_dropdu[lanlot_cols[0]], df_dropdu[lanlot_cols[1]])])
        lanlot_region = gpd.sjoin(lanlot.reset_index().rename(columns={0: 'geometry'}).set_crs('epsg:4326').to_crs('EPSG:2381'),
                                  gdf_region.loc[:, [name, 'geometry']])
        df_dropdu = df_dropdu.join(lanlot_region.set_index('index').loc[:, name])
    df = df.merge(df_dropdu.loc[:, lanlot_cols + region_name], how='left', on=lanlot_cols)
    return df


def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">点击下载 {file_label}</a>'
    return href

st.title('区域划分工具')
st.header('1、文件导入')
file_obj = st.file_uploader('上传一个表格', type=['csv', 'xlsx', 'xls'])

if file_obj:
    df_source = read_df(file_obj)
    region_dict = layer_check('qgis')

    table_name = st.selectbox(
        "图层选择",
        region_dict.keys()
    )
    geo_df = gpd.read_file(f'.//qgis//{region_dict[table_name]}.gpkg', encoding='utf-8')
    fig, ax = plt.subplots()
    ax = geo_df.plot(ax=ax, column=table_name, cmap='Spectral')
    st.pyplot(fig)
    if not st.button('执行'):
        st.stop()
    else:
        result = region_division(df_source, region_dict, table_name)
        result.to_csv(f'区域划分结果-{table_name}.csv', index=False, encoding='utf_8_sig')
        st.header('2、结果展示')
        st.dataframe(result.head(20))
        st.header('3、地图显示')
        st.map(result.rename(columns={'经度': 'lon', '纬度': 'lat'}).head(20))
        st.header('4、结果下载')
        file_path, file_label = f'区域划分结果-{table_name}.csv', '区域划分结果'
        st.markdown(get_binary_file_downloader_html(file_path, file_label), unsafe_allow_html=True)
