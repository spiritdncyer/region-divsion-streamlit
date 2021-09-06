import streamlit as st
# import os
import time
import re
# import math
import chardet
# import base64
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
            st.sidebar.write('、'.join(region_name) + '已划分')
            st.sidebar.write(f'{step}耗时：{float(time.time() - start):.3f}秒')
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

    if df is not None and {'经度', '纬度'}.issubset(set(list(df))):
        df = df.pipe(clean_lotlan).astype(lonlat_float_format)
    else:
        raise ValueError('当前表格格式错误')
    return df


def layer_check(geofolder):
    # gdf_list = []
    try:
        dictionary = dict(pd.read_csv(f'.//{geofolder}//图层信息.csv', encoding='gb18030').loc[:, ['字段名称', '图层名称']].values)
        key_list = dictionary.keys()
        file_extension = 'shp' if geofolder == 'mapinfo' else 'gpkg'
        for index, name in enumerate(key_list):
            gdf = gpd.read_file(f'.//{geofolder}//{dictionary[name]}.{file_extension}', encoding='utf-8')
            if name not in list(gdf):
                raise ValueError(f'图层字段<{name}>不在图层<{dictionary[name]}.{file_extension}>中')
            else:
                dictionary[name] = [dictionary[name]]
                dictionary.setdefault(name, []).append(gdf)
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
        # gdf_region = gpd.read_file(f'.//{geofolder}//{region_dictionary[name][0]}.{file_extension}', encoding='utf-8')
        gdf_region = region_dictionary[name][1]
        gdf_region = gdf_region.to_crs('EPSG:2381') if gdf_region.crs is None else gdf_region.to_crs('EPSG:2381')
        lanlot = gpd.GeoSeries([Point(x, y) for x, y in zip(df_dropdu[lanlot_cols[0]], df_dropdu[lanlot_cols[1]])])
        lanlot_region = gpd.sjoin(lanlot.reset_index().rename(columns={0: 'geometry'}).set_crs('epsg:4326').to_crs('EPSG:2381'),
                                  gdf_region.loc[:, [name, 'geometry']])
        df_dropdu = df_dropdu.join(lanlot_region.set_index('index').loc[:, name])
        my_bar = st.progress(0)
        my_bar.progress((index + 1) / len(region_name))
    df = df.merge(df_dropdu.loc[:, lanlot_cols + region_name], how='left', on=lanlot_cols)
    return df


def text_to_df(text):
    search_result = re.findall(r'(?P<lon>1[12][0-9].\d+)[\s,，]*(?P<lat>3[12].\d+)', text)
    if search_result:
        point = {}
        for lon_lat in search_result:
            point.setdefault('经度', []).append(float(lon_lat[0]))
            point.setdefault('纬度', []).append(float(lon_lat[1]))
        return pd.DataFrame(data=point)


def ouput(df):
    return df.to_csv().encode('utf-8-sig')


# def get_binary_file_downloader_html(bin_file, file_label='File'):
#     with open(bin_file, 'rb') as f:
#         data = f.read()
#     bin_str = base64.b64encode(data).decode()
#     href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">点击下载 {file_label}</a>'
#     return href


st.title('区域划分工具')

st.header('1、图层展示')
region_dict = layer_check('qgis')
st.sidebar.header('图层信息')
table_name = st.sidebar.multiselect(
    "选择图层",
    region_dict.keys(),
    default=['区县', '三方区域'],
)
plt.rcParams['font.size'] = 5
fig_list, num_fig = [], len(table_name)
num_subplot, last_col = (num_fig - 1) // 3 + 1, (num_fig - 1) % 3 + 1

for n in range(num_subplot):
    if n == num_subplot - 1:
        fig, ax = plt.subplots(1, last_col, sharex='all', sharey='all', figsize=(2 * last_col, 1.8))
        plt.locator_params('x', nbins=10)
        # plt.tick_params(labelsize=3)
        for i, field_name in enumerate(table_name[3 * n:]):
            # geo_df = gpd.read_file(f'.//qgis//{region_dict[field_name]}.gpkg', encoding='utf-8')
            geo_df = region_dict[field_name][1]
            ax_i = ax if last_col == 1 else ax[i % last_col]
            ax_i.set_xlim(119.1, 120.3)
            ax_i.set_ylim(31.1, 32.1)
            geo_df.plot(ax=ax_i, column=field_name, cmap='Spectral')
        fig_list.append(fig)
    else:
        fig, ax = plt.subplots(1, 3, sharex='all', sharey='all', figsize=(6, 1.8))
        plt.locator_params('x', nbins=10)
        # plt.tick_params(labelsize=3)
        for i, field_name in enumerate(table_name[3 * n:3 * (n + 1)]):
            # geo_df = gpd.read_file(f'.//qgis//{region_dict[field_name]}.gpkg', encoding='utf-8')
            geo_df = region_dict[field_name][1]
            ax_i = ax[i % 3]
            ax_i.set_xlim(119.1, 120.3)
            ax_i.set_ylim(31.1, 32.1)
            geo_df.plot(ax=ax_i, column=field_name, cmap='Spectral')
        fig_list.append(fig)

if table_name:
    name_list = '、'.join(table_name)
    st.write(f'选择的图层为：{name_list}')
    for fig in fig_list:
        st.pyplot(fig)

    st.header('2、数据选择')
    st.sidebar.header('输入模式')
    input_mode = st.sidebar.radio('输入方式选择', ('手动输入', '文件导入'))
    if input_mode == '手动输入':
        st.write('数据选择模式：手动输入')
        # st.sidebar.subheader('2.1手动输入')
        input_text = st.sidebar.text_input(
            '输入经纬度',
            value='例如：119.934	31.8528 119.939	31.84',
            help='输入经纬度数据，可直接复制粘贴excel表格中的经度、纬度2列数据'
        )
        df_source = text_to_df(input_text)
        st.write('数据源：')
        st.table(df_source)

        if not st.sidebar.button('执行区域划分'):
            st.stop()
        else:
            st.sidebar.header('输出结果')
            result = region_division(df_source, region_dict, table_name)

            st.header('3、输出表格')
            st.table(result)

            st.header('4、地图展示')
            st.map(result.rename(columns={'经度': 'lon', '纬度': 'lat'}))

            st.sidebar.header('数据下载')
            st.sidebar.download_button(
                label='下载结果',
                data=ouput(result),
                file_name=f'区域划分结果-{name_list}.csv',
                mime='text/csv',
            )
    else:
        st.write('数据选择模式：文件导入')
        # st.subheader('2.2文件导入')
        file_obj = st.sidebar.file_uploader(
            '上传一个表格',
            type=['csv', 'xlsx', 'xls'],
            help='上传文件格式为csv、xlsx、xls，需包含经度、纬度2列数据'
        )
        rows = st.sidebar.slider(
            '选择数据源显示行数',
            0, 50, 20, 10
        )
        sample_rows = st.sidebar.slider(
            '选择结果采样行数',
            0, 50, 20, 10
        )
        if file_obj:
            df_source = read_df(file_obj)
            st.write(f'数据源（前{rows}行）：')
            st.table(df_source.head(rows))

            if not st.sidebar.button('执行区域划分'):
                st.stop()
            else:
                st.sidebar.header('输出结果')
                result = region_division(df_source, region_dict, table_name)

                st.header('3、输出表格')
                st.write(f'随机采样{sample_rows}行')
                df_sample = result.sample(sample_rows)
                st.table(df_sample)

                st.header('4、地图展示')
                st.map(df_sample.rename(columns={'经度': 'lon', '纬度': 'lat'}))

                st.sidebar.header('数据下载')
                st.sidebar.download_button(
                    label='下载结果',
                    data=ouput(result),
                    file_name=f'区域划分结果-{name_list}.csv',
                    mime='text/csv',
                )
                # file_path, file_label = f'区域划分结果-{table_name}.csv', '区域划分结果'
                # st.markdown(get_binary_file_downloader_html(file_path, file_label), unsafe_allow_html=True)
