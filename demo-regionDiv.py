import streamlit as st
import time
import re
import chardet
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from functools import wraps
from shapely.geometry import Point


def main():
    if 'run' not in st.session_state:
        st.session_state.run = 0

    if 'layer_selector' not in st.session_state:
        st.session_state.layer_selector = 0

    st.title('区域划分工具')

    region_dict = read_layer_and_check('qgis')
    selected_name, submit = layer_selector(region_dict)
    input_mode = input_mode_selector()

    if st.session_state.layer_selector:
        if input_mode == '文本输入':
            run_manual_input(region_dict, selected_name)
        else:
            run_file_input(region_dict, selected_name)

    st.write('执行次数为：', st.session_state.run)


@st.cache
def read_layer_and_check(geofolder):
    try:
        dictionary = dict(pd.read_csv(f'.//{geofolder}//图层信息.csv', encoding='gb18030').loc[:, ['字段名称', '图层名称']].values)
        key_list = dictionary.keys()
        file_extension = 'shp' if geofolder == 'mapinfo' else 'gpkg'
        for index, name in enumerate(key_list):
            gdf = gpd.read_file(f'.//{geofolder}//{dictionary[name]}.{file_extension}', encoding='utf-8')
            if name not in list(gdf):
                st.error(f'图层字段<{name}>不在图层<{dictionary[name]}.{file_extension}>中')
            else:
                dictionary[name] = [dictionary[name]]
                dictionary.setdefault(name, []).append(gdf)
        return dictionary
    except IOError:
        st.error(f'找不到图层信息')


def layer_selector(region_dictionary):
    st.header('1、图层展示')
    with st.form(key='selector'):
        # st.subheader('图层信息选择')
        region_name = st.multiselect(
            "请选择图层",
            region_dictionary.keys(),
            default=['区县', '三方区域', '规划区域'],
        )
        submit = st.form_submit_button(label='确认', on_click=layer_selector_counter)
    figure = layer_ploting(region_dictionary, region_name, 3)
    if region_name:
        name_list = '、'.join(region_name)
        st.write(f'选择的图层为：{name_list}')
        st.pyplot(figure)
    return region_name, submit


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def layer_ploting(region_dictionary, region_name, fig_cols):
    plt.rcParams['font.size'] = 5
    num_fig = len(region_name)
    if num_fig > 0:
        nrows = (num_fig - 1) // fig_cols + 1
        fig, ax = plt.subplots(nrows, fig_cols, figsize=(3 * fig_cols, 3 * nrows))
        for i, field_name in enumerate(region_name):
            geo_df = region_dictionary[field_name][1]
            if nrows == 1:
                ax_i = ax[i]
            else:
                ax_rows, ax_cols = i // fig_cols, i % fig_cols
                ax_i = ax[ax_rows][ax_cols]
            ax_i.set_xlim(119.1, 120.3)
            ax_i.set_ylim(31.1, 32.1)
            geo_df.plot(ax=ax_i, column=field_name, cmap='Spectral')
        # 去掉坐标轴
        mod_num = num_fig % fig_cols
        if mod_num != 0:
            if nrows == 1:
                for n in range(mod_num, fig_cols):
                    ax[n].axis('off')
            else:
                for n in range(mod_num, fig_cols):
                    ax[nrows - 1][n].axis('off')
    else:
        fig, ax = plt.subplots()
        ax.axis('off')
    # st.write("Cache miss: layer_ploting")
    return fig


def input_mode_selector():
    st.header('2、数据选择')
    st.sidebar.header('输入模式选择')
    return st.sidebar.radio(
        '请选择输入方式',
        ('文件导入', '文本输入'),
        help='首次执行请先在图层选择处点击确认。'
    )


def run_manual_input(region_dictionary, region_name):
    st.write('数据选择模式：文本输入')
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
        result = region_division(df_source, region_dictionary, region_name)

        st.header('3、输出表格')
        st.table(result)

        st.header('4、地图展示')
        st.map(result.rename(columns={'经度': 'lon', '纬度': 'lat'}))

        st.sidebar.header('数据下载')
        name_list = '、'.join(region_name)
        st.sidebar.download_button(
            label='下载结果',
            data=ouput(result),
            file_name=f'区域划分结果-{name_list}.csv',
            mime='text/csv',
        )


def text_to_df(text):
    search_result = re.findall(r'(?P<lon>1[12][0-9].\d+)[\s,，]*(?P<lat>3[12].\d+)', text)
    if search_result:
        point = {}
        for lon_lat in search_result:
            point.setdefault('经度', []).append(float(lon_lat[0]))
            point.setdefault('纬度', []).append(float(lon_lat[1]))
        return pd.DataFrame(data=point)
    else:
        st.error('输入格式错误')


def run_file_input(region_dictionary, region_name):
    st.write('数据选择模式：文件导入')
    file_obj = st.sidebar.file_uploader(
        '上传一个表格',
        type=['csv', 'xlsx', 'xls'],
        help='上传文件格式为csv、xlsx、xls，需包含表头为经度、纬度的2列数据',
    )

    if file_obj:
        # 清理数据、执行区域划分
        df_source = read_df(file_obj)
        if df_source is None:
            st.stop()
        st.sidebar.header('输出结果')
        result = region_division(df_source, region_dictionary, region_name)
        # 显示数据源
        render_rows = 10 if df_source.shape[0] >= 10 else df_source.shape[0] // 5 * 5
        rows = st.sidebar.slider(
            '选择数据源显示行数',
            0, 50, render_rows, 5
        )
        st.write(f'数据源（前{rows}行）：')
        st.dataframe(df_source.head(rows))
        # 结果采样
        st.header('3、输出表格')
        sample_rows = st.sidebar.slider(
            '选择结果采样行数',
            0, 50, render_rows, 5
        )
        st.write(f'随机采样{sample_rows}行：')
        df_sample = result.sample(sample_rows)
        st.dataframe(df_sample)
        # 结果可视化
        st.header('4、统计图表')
        summary, rail_data = reslut_summary(result, region_name)
        fig_list = summary_ploting(summary, rail_data)
        for figure in fig_list:
            st.pyplot(figure)
        # 数据下载
        st.sidebar.header('数据下载')
        name_list = '、'.join(region_name)
        st.sidebar.download_button(
            label='下载明细结果',
            data=ouput(result),
            file_name=f'区域划分结果-{name_list}.csv',
            mime='text/csv',
            help='区域划分的明细数据',
            )
        st.sidebar.download_button(
            label='下载统计结果',
            data=output_summary(summary),
            file_name=f'区域划分统计结果-{name_list}.csv',
            mime='text/csv',
            help='统计每个图层各个区域的数量',
        )


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


LONLAT_STR_FORMAT = {'经度': 'string', '纬度': 'string'}
LONLAT_FLOAT_FORMAT = {'经度': 'float64', '纬度': 'float64'}


def df_clean(df):
    if {'经度', '纬度'}.issubset(set(list(df))):
        return df.pipe(clean_lotlan).astype(LONLAT_FLOAT_FORMAT)
    else:
        st.error('当前表格格式错误')
        st.sidebar.error('当前表格格式错误')


def clean_lotlan(df_cell):
    for col_name in list(df_cell.loc[:, ['经度', '纬度']]):
        df_cell[col_name] = df_cell.astype({col_name: 'string'})[col_name].str.replace(r'\s', '', regex=True)
    df_cell_split_list = df_cell['经度'].str.contains('/')
    df_cell_split = df_cell[df_cell_split_list]
    if not df_cell_split.empty:
        df_comb = pd.DataFrame([], index=df_cell_split.index)
        for col_name in list(df_cell_split.loc[:, ['经度', '纬度']]):
            df_comb = pd.concat([df_comb, (df_cell_split[col_name].str.split('/', expand=True)
                                           .stack().reset_index(level=1).rename(columns={0: col_name}))], axis=1)
        df_cell = pd.concat([df_cell[~df_cell_split_list],
                             df_cell_split.iloc[:, :3].join(df_comb.drop(['level_1'], axis=1))]).reset_index(drop=True)
    return df_cell


@st.cache(suppress_st_warning=True)
def read_df(file):
    f_ext = file.name.split('.')[1]
    df = None
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
            st.error('文件编码错误')
    elif f_ext in ['xlsx', 'xls']:
        df = pd_read(file, f_ext)
    else:
        st.error('文件格式错误')
    # st.write("Cache miss:read_df")
    return df


def pd_read(file, extension, encode_n=None):
    try:
        if extension == 'csv':
            return pd.read_csv(file, dtype=LONLAT_STR_FORMAT, encoding=encode_n, low_memory=False)
        elif extension in ['xlsx', 'xls']:
            return pd.read_excel(file, dtype=LONLAT_STR_FORMAT)
        else:
            st.error('文件格式错误')
    except ValueError:
        st.error('文件读取错误')


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
@time_costing('区域划分')
def region_division(df, region_dictionary, region_name):
    lanlot_cols = ['经度', '纬度']
    df = df_clean(df)
    if isinstance(region_name, str):
        region_name = [region_name]
    elif isinstance(region_name, list):
        pass
    else:
        st.error('错误：区域名称错误')
    df_dropdu = df.drop_duplicates(subset=lanlot_cols).reset_index(drop=True)
    my_bar = st.sidebar.progress(0)
    for index, name in enumerate(region_name):
        gdf_region = region_dictionary[name][1]
        gdf_region = gdf_region.to_crs('EPSG:2381') if gdf_region.crs is None else gdf_region.to_crs('EPSG:2381')
        lanlot = gpd.GeoSeries([Point(x, y) for x, y in zip(df_dropdu[lanlot_cols[0]], df_dropdu[lanlot_cols[1]])])
        lanlot_region = gpd.sjoin(lanlot.reset_index().rename(columns={0: 'geometry'})
                                  .set_crs('epsg:4326').to_crs('EPSG:2381'), gdf_region.loc[:, [name, 'geometry']])
        df_dropdu = df_dropdu.join(lanlot_region.set_index('index').loc[:, name])
        my_bar.progress((index + 1) / len(region_name))
    df = df.merge(df_dropdu.loc[:, lanlot_cols + region_name], how='left', on=lanlot_cols)
    # st.write("Cache miss: region_division")
    run_counter()
    return df


def run_counter():
    st.session_state.run += 1


def layer_selector_counter():
    st.session_state.layer_selector += 1


def ouput(df):
    return df.to_csv(index=False).encode('utf-8-sig')


def output_summary(summary):
    df_summary = pd.DataFrame([])
    for key in summary.keys():
        df_summary = pd.concat([df_summary, summary[key]], axis=1)
    return df_summary.to_csv(index=False).encode('utf-8-sig')


@st.cache(suppress_st_warning=True)
def reslut_summary(df, region_name):
    for name in region_name:
        if name == '规划区域':
            df['规划区域'] = df['规划区域'].fillna('农村')
        elif name == '网格区域':
            df['网格区域'] = df['网格区域'].fillna('网格外')
        elif name == '高铁周边':
            df['高铁周边'] = df['高铁周边'].fillna('铁路外')
        else:
            df[name] = df[name].fillna('其他')

    county_order = ['天宁', '钟楼', '武进', '新北', '经开', '金坛', '溧阳', '其他']
    third_party_order = ['华星', '华苏-武进', '华苏-金坛', '华苏-溧阳', '其他']
    planning_region_order = ['主城区', '一般城区', '县城', '乡镇', '农村']
    grid_order = ['网格内', '网格边界200米', '网格外']
    rail_surrounding_order = ['京沪周边500米', '京沪周边1.5公里', '沪宁周边500米', '沪宁周边1.5公里', '宁杭周边500米', '宁杭周边1.5公里', '铁路外']
    tag_order = ['主城区', '县城', '其他']

    name_list = ['区县', '三方区域', '规划区域', '网格区域', '高铁周边', '标签区域']
    order_list = [county_order, third_party_order, planning_region_order, grid_order, rail_surrounding_order, tag_order]

    region_order_dict = dict(zip(name_list, order_list))
    summary = {}
    for name in region_name:
        summary[name] = (df.groupby(name)['ECGI'].count().reset_index(name='数量')
                         .assign(temp=lambda x: x[name].astype('category').cat.set_categories(region_order_dict[name]))
                         .sort_values(by=['temp'], ignore_index=True).drop('temp', axis=1))
    rail_data = summary.pop('高铁周边') if summary.get('高铁周边') is not None else None
    # st.write("Cache miss: reslut_summary")
    return summary, rail_data


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def summary_ploting(summary, rail_data):
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig_list = []
    region_name = list(summary.keys())
    num_name = len(region_name)
    nrows = (num_name - 1) // 4 + 1
    if num_name > 0:
        fig, ax = plt.subplots(nrows, 2, figsize=(10, 4.8 * nrows))
        # 每2组画环形饼图（剔除高铁）
        for index in range(0, num_name, 2):
            name_1 = region_name[index]
            name_2 = region_name[index + 1] if index < num_name - 1 else None
            if nrows == 1:
                ax_i = ax[index // 2]
            else:
                ax_rows, ax_cols = index // 2 // 2, index // 2 % 2
                ax_i = ax[ax_rows][ax_cols]
            if name_2 is not None:
                size = 0.3
                labels_1, vals_1 = summary[name_1][name_1].to_list(), summary[name_1]['数量'].values
                labels_2, vals_2 = summary[name_2][name_2].to_list(), summary[name_2]['数量'].values
                num_label_1, num_label_2 = len(labels_1), len(labels_2)
                cmap = plt.get_cmap("tab20c")

                if num_label_1 <= num_label_2:
                    outer_labels, outer_vals = labels_1, vals_1
                    inner_labels, inner_vals = labels_2, vals_2
                    outer_colors = cmap(tab20c_color_array(num_label_1, 'outer'))
                    inner_colors = cmap(tab20c_color_array(num_label_2, 'inner'))
                else:
                    outer_labels, outer_vals = labels_2, vals_2
                    inner_labels, inner_vals = labels_1, vals_1
                    outer_colors = cmap(tab20c_color_array(num_label_2, 'outer'))
                    inner_colors = cmap(tab20c_color_array(num_label_1, 'inner'))

                wedges1, texts1, autotexts1 = ax_i.pie(
                    inner_vals, radius=1 - size, labels=inner_labels, colors=inner_colors,
                    autopct=lambda pct: pct_func(pct, inner_vals), pctdistance=0.75, labeldistance=0.3,
                    startangle=90, wedgeprops=dict(width=size, edgecolor='w')
                )
                wedges2, texts2, autotexts2 = ax_i.pie(
                    outer_vals, radius=1, labels=outer_labels, colors=outer_colors,
                    autopct=lambda pct: pct_func(pct, outer_vals), pctdistance=0.85,
                    startangle=90, wedgeprops=dict(width=size, edgecolor='w')
                )
                plt.setp(autotexts1, size=10, weight="bold", color="w")
                plt.setp(autotexts2, size=10, weight="bold", color="w")
                plt.setp(texts1, size=10, color="k")
                plt.setp(texts2, size=10, color="k")
                ax_i.set(aspect="equal")
            else:
                # 单独剩一个画传统饼图
                labels_1, vals_1 = summary[name_1][name_1].to_list(), summary[name_1]['数量'].values
                num_label_1 = len(labels_1)
                cmap = plt.get_cmap("tab20c")
                outer_colors = cmap(tab20c_color_array(num_label_1, 'inner'))
                wedges, texts, autotexts = ax_i.pie(vals_1, radius=1, labels=labels_1, colors=outer_colors,
                                                    autopct=lambda pct: pct_func(pct, vals_1), startangle=90)
                plt.setp(autotexts, size=10, weight="bold", color="w")
                plt.setp(texts, size=10, weight="bold", color="k")
                ax_i.set(aspect="equal")
        plt.axis('off')
        fig_list.append(fig)
        # 画高铁复合饼图
        if rail_data is not None:
            fig = plt.figure(figsize=(10, 4.8))
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            fig.subplots_adjust(wspace=0)

            merged_label = ['高铁周边', '铁路外']
            df_rail = rail_data.query('高铁周边 != "铁路外"')
            merged_val = [df_rail['数量'].sum(), rail_data.query('高铁周边 == "铁路外"')['数量'].sum()]
            angle = -180 * merged_val[0] / merged_val[1]
            explode = [0.1, 0]
            cmap = plt.get_cmap("tab20c")
            merged_colors = cmap([4, 0])

            wedges1, texts1, autotexts1 = ax1.pie(merged_val, radius=1, labels=merged_label, colors=merged_colors,
                                                  autopct=lambda pct: pct_func(pct, merged_val),
                                                  startangle=angle, explode=explode)
            plt.setp(autotexts1, size=10, weight="bold", color="w")
            plt.setp(texts1, size=12, color="k")

            detail_label, detail_val = df_rail['高铁周边'].to_list(), df_rail['数量'].values
            num_label = len(detail_label)
            cmap = plt.get_cmap("tab20c")
            detail_colors = cmap(tab20c_color_array(num_label, 'inner'))

            r2 = 0.8
            wedges2, texts2, autotexts2 = ax2.pie(detail_val, radius=r2, labels=detail_label, colors=detail_colors,
                                                  autopct=lambda pct: pct_func(pct, detail_val),
                                                  startangle=90, counterclock=False)
            plt.setp(autotexts2, size=10, weight="bold", color="w")
            plt.setp(texts2, size=10, color="k")

            # 饼图边缘的数据
            theta1 = ax1.patches[0].theta1
            theta2 = ax1.patches[0].theta2
            center = ax1.patches[0].center
            r = ax1.patches[0].r
            width = 0.2
            # 上边缘的连线
            x = r * np.cos(np.pi / 180 * theta2) + center[0]
            y = r * np.sin(np.pi / 180 * theta2) + center[1]
            con_a = ConnectionPatch(xyA=(-width / 2, r2), xyB=(x, y), coordsA='data', coordsB='data', axesA=ax2, axesB=ax1)
            # 下边缘的连线
            x = r * np.cos(np.pi / 180 * theta1) + center[0]
            y = r * np.sin(np.pi / 180 * theta1) + center[1]
            con_b = ConnectionPatch(xyA=(-width / 2, -r2), xyB=(x, y), coordsA='data', coordsB='data', axesA=ax2, axesB=ax1)

            for con in [con_a, con_b]:
                con.set_linewidth(1)  # 连线宽度
                con.set_color = ([0, 0, 0])  # 连线颜色
                ax2.add_artist(con)  # 添加连线

            fig_list.append(fig)
    else:
        pass
    # st.write("Cache miss: summary_ploting")
    return fig_list


def pct_func(pct, allvals):
    absolute = int(round(pct/100.*np.sum(allvals)))
    return "{:d}\n{:.1f}%".format(absolute, pct)


def tab20c_color_array(num_label, outer_or_inner):
    array = np.empty((0, 5))
    if outer_or_inner == 'outer':
        outer_layer_num = (num_label - 1) // 5 + 1
        for i in range(outer_layer_num):
            array = np.append(array, np.arange(5) * 4 + i)
        array = np.sort(array).astype(int)
    elif outer_or_inner == 'inner':
        inner_layer_num = (num_label - 1) // 10 + 1
        for i in range(inner_layer_num):
            if i == 0:
                array = np.append(np.arange(5) * 4 + 1, np.arange(5) * 4 + 2)
            else:
                array = np.append(array, np.arange(5) * 4 + i + 2)
    return np.sort(array)


if __name__ == "__main__":
    main()
