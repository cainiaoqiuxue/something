from pyecharts.charts import Bar, Pie, ThemeRiver, WordCloud, Line, HeatMap, Graph
from pyecharts import options as opts
from pyecharts.commons.utils import JsCode
from pyecharts.globals import ThemeType


class PlotUtils:

    @staticmethod
    def plot_bar(res_dic, title='柱状图'):
        names = list(res_dic.keys())
        values = list(res_dic.values())
        bar = Bar()
        bar.add_xaxis(names)
        bar.add_yaxis("数量", values)
        bar.set_global_opts(title_opts=opts.TitleOpts(title=title))
        # bar.render("fruit_bar_chart.html")
        # bar.render_notebook()
        return bar

    @staticmethod
    def plot_base_pie(res_dic, title='基础饼图'):
        names = list(res_dic.keys())
        values = list(res_dic.values())
        pie = Pie()
        pie.add("", [list(z) for z in zip(names, values)])
        pie.set_global_opts(title_opts=opts.TitleOpts(title=title))
        pie.set_series_opts(
                tooltip_opts=opts.TooltipOpts(
                    trigger="item", formatter="{b}: {c} ({d}%)"
                )
                )
        # pie.render("fruit_pie_chart.html")
        # pie.render_notebook()
        return pie


    @staticmethod
    def plot_multi_pie(inner_dic, outer_dic, source='访问来源'):
        inner_x_data = list(inner_dic.keys())
        inner_y_data = list(inner_dic.values())
        inner_data_pair = [list(z) for z in zip(inner_x_data, inner_y_data)]

        outer_x_data = list(outer_dic.keys())
        outer_y_data = list(outer_dic.values())
        outer_data_pair = [list(z) for z in zip(outer_x_data, outer_y_data)]

        pie = (
            Pie()
            # .add(
            #     series_name=source,
            #     data_pair=inner_data_pair,
            #     radius=[0, "30%"],
            #     label_opts=opts.LabelOpts(position="inner"),
            # )
            .add(
                series_name=source,
                radius=["40%", "55%"],
                data_pair=outer_data_pair,
                label_opts=opts.LabelOpts(
                    position="outside",
                    formatter="{a|{a}}{abg|}\n{hr|}\n {b|{b}: }{c}  {per|{d}%}  ",
                    background_color="#eee",
                    border_color="#aaa",
                    border_width=1,
                    border_radius=4,
                    rich={
                        "a": {"color": "#999", "lineHeight": 22, "align": "center"},
                        "abg": {
                            "backgroundColor": "#e3e3e3",
                            "width": "100%",
                            "align": "right",
                            "height": 22,
                            "borderRadius": [4, 4, 0, 0],
                        },
                        "hr": {
                            "borderColor": "#aaa",
                            "width": "100%",
                            "borderWidth": 0.5,
                            "height": 0,
                        },
                        "b": {"fontSize": 16, "lineHeight": 33},
                        "per": {
                            "color": "#eee",
                            "backgroundColor": "#334455",
                            "padding": [2, 4],
                            "borderRadius": 2,
                        },
                    },
                ),
            )
            .set_global_opts(legend_opts=opts.LegendOpts(is_show=False),)
            # .set_global_opts(legend_opts=opts.LegendOpts(type_="scroll", pos_left="80%", orient="vertical"),)
            .set_series_opts(
                tooltip_opts=opts.TooltipOpts(
                    trigger="item", formatter="{a} <br/>{b}: {c} ({d}%)"
                )
            )

        )
        return pie

    @staticmethod
    def plot_them_river(labels, data):
        x_data = labels
        y_data = data

        pic =  (
            ThemeRiver()
            .add(
                series_name=x_data,
                data=y_data,
                singleaxis_opts=opts.SingleAxisOpts(
                    pos_top="50", pos_bottom="50", type_="time"
                ),
            )
            .set_global_opts(
                tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="line")
            )
        )
        return pic

    @staticmethod
    def plot_wordcloud(data, name='词云图'):
        pic = (
        WordCloud()
        .add(series_name=name, data_pair=data, word_size_range=[6, 66])
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title=name, title_textstyle_opts=opts.TextStyleOpts(font_size=23)
            ),
            tooltip_opts=opts.TooltipOpts(is_show=True),
            )
        )
        return pic


    @staticmethod
    def plot_stack_bar(data, name='堆叠柱状图'):
        c = (
            Bar()
            .add_xaxis(['官媒', '民媒'])
            .add_yaxis("情感正向", data[0], stack="stack1")
            .add_yaxis("情感中立", data[1], stack="stack1")
            .add_yaxis("情感负向", data[2], stack="stack1")
            .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
            .set_global_opts(title_opts=opts.TitleOpts(title=name))
        )
        return c

    @staticmethod
    def plot_stack_pie(data, source='情感倾向'):

        inner_x_data = ["官媒正向", "官媒中性", "官媒负向"]
        inner_y_data = [data[1], data[0], data[2]]
        inner_data_pair = [list(z) for z in zip(inner_x_data, inner_y_data)]

        outer_x_data = ["网媒正向", "网媒中性", "网媒负向"]
        outer_y_data = [data[4], data[3], data[5]]
        outer_data_pair = [list(z) for z in zip(outer_x_data, outer_y_data)]

        c = (
            Pie()
            .add(
                series_name=source,
                data_pair=inner_data_pair,
                radius=[0, "30%"],
                label_opts=opts.LabelOpts(position="inner"),
            )
            .add(
                series_name=source,
                radius=["40%", "55%"],
                data_pair=outer_data_pair,
                label_opts=opts.LabelOpts(
                    position="outside",
                    background_color="#eee",
                    border_color="#aaa",
                    border_width=1,
                    border_radius=4,
                ),
            )
            .set_global_opts(legend_opts=opts.LegendOpts(pos_left="left", orient="vertical"))
            .set_series_opts(
                tooltip_opts=opts.TooltipOpts(
                    trigger="item", formatter="{a} <br/>{b}: {c} ({d}%)"
                )
            )
        )
        return c


    @staticmethod
    def plot_line(data, name='折线图'):
        date = data[0]
        value = data[1]
        a = value.keys()
        b = value.values()
        a = list(a)
        b = list(b)
        c = (
            Line()
            .add_xaxis(list(date))
            .add_yaxis(a[0], b[0])
            .add_yaxis(a[1], b[1])
            .add_yaxis(a[2], b[2])
            .add_yaxis(a[3], b[3])
            .add_yaxis(a[4], b[4])
            .add_yaxis(a[5], b[5])
            .set_global_opts(title_opts=opts.TitleOpts(title=name))
        )
        return c

    @staticmethod
    def plot_heatmap(row, col, data, name='情感倾向'):
        c = (
            HeatMap()
            .add_xaxis(xaxis_data=row)
            .add_yaxis(
                series_name=name,
                yaxis_data=col,
                value=data,
                label_opts=opts.LabelOpts(
                    is_show=True, color="#fff", position="bottom", horizontal_align="50%"
                ),
            )
            .set_series_opts()
            .set_global_opts(
                legend_opts=opts.LegendOpts(is_show=False),
                xaxis_opts=opts.AxisOpts(
                    type_="category",
                    splitarea_opts=opts.SplitAreaOpts(
                        is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
                    ),
                ),
                yaxis_opts=opts.AxisOpts(
                    type_="category",
                    splitarea_opts=opts.SplitAreaOpts(
                        is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
                    ),
                ),
                visualmap_opts=opts.VisualMapOpts(
                    min_=0, max_=10, is_calculable=True, orient="horizontal", pos_left="center"
                ),
            )
        )
        return c
    
    @staticmethod
    def plot_base_pie_with_word(res_dic, title='基础饼图'):
        code = 'aaa'
        formats = """
            function (params) {
                var message = '';
                switch (params.name) {
                    case '情感正向':
                        message = '【华夏时报】A股力挺新疆棉?纺织板块掀涨停潮,国产服装品牌迎来爆发期<br/>【新浪科技】商务部：纯白无瑕的新疆棉花不容任何势力抹黑玷污<br/>【网易新闻】暖心又文艺！新疆演员陈建斌为家乡棉花写小诗<br/>......';
                        break;
                    case '情感中立':
                        message = '【人民资讯】这是真正的新疆<br/>【中時新聞網】新疆棉風暴擴大 前外交官4字解析內幕<br/>【天山植保】2021新疆棉花质量补贴四大变化<br/>......';
                        break;
                    case '情感负向':
                        message = '【中国经济网】新疆棉花中国自己还不够用 H&M遭抵制是自食其果<br/>【微博】明星集体支持新疆棉花！王一博率先解约，还有两位“顶流”态度不明<br/>【深港在线综合】台艺人挺新疆棉绿媒坐不住 绿媒行径引众怒！！<br/>......';
                        break;
                    default:
                        message = 'Unknown category';
                }
                return message;
            }
            """.replace('SEPD', code)
        names = list(res_dic.keys())
        values = list(res_dic.values())
        pie = Pie()
        pie.add("", [list(z) for z in zip(names, values)], 
                selected_mode='single', label_opts=opts.LabelOpts(formatter="{b}: {d}%"),)
        pie.set_global_opts(title_opts=opts.TitleOpts(title=title), legend_opts=opts.LegendOpts(is_show=False),)
        pie.set_series_opts(
                tooltip_opts=opts.TooltipOpts(
                trigger="item",
                formatter=JsCode(formats)
            )
                )
        return pie
    
    @staticmethod
    def plot_graph(nodes, links, word_freq):
        graph = Graph()
        graph.add("", 
                nodes=[{"name": word, "value": word_freq[word]} for word in nodes],
                links=links,
                repulsion=8000)
        graph.set_global_opts(title_opts=opts.TitleOpts(title="词语共现关系图"))
        return graph
    
    @staticmethod
    def plot_line_with_them(x, y, special_data):
        c = (
            Line()
            .add_xaxis(x)
            .add_yaxis(
                "舆情事件",
                y,
                markpoint_opts=opts.MarkPointOpts(
                    data=[opts.MarkPointItem(name=i[1], coord=[x[i[0]], y[i[0]]], value='事件{}'.format(j + 1)) for j, i in enumerate(special_data)]
                ),
            )
            .set_global_opts(title_opts=opts.TitleOpts(title="趋势图"))
        )
        return c
    
    @staticmethod
    def plot_stack_pie_split(data):
        inner_x_data = ["官媒正向", "官媒中性", "官媒负向"]
        inner_y_data = [data[1], data[0], data[2]]
        inner_data_pair = [list(z) for z in zip(inner_x_data, inner_y_data)]

        outer_x_data = ["网媒正向", "网媒中性", "网媒负向"]
        outer_y_data = [data[4], data[3], data[5]]
        outer_data_pair = [list(z) for z in zip(outer_x_data, outer_y_data)]

        c = (
            Pie()
            .add(
                "",
                inner_data_pair,
                center=["20%", "50%"],
                radius=[60, 80],
            )
            .add(
                "",
                outer_data_pair,
                center=["55%", "50%"],
                radius=[60, 80],
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title="不同媒体情感分布"),
                legend_opts=opts.LegendOpts(
                    type_="scroll", pos_top="20%", pos_left="80%", orient="vertical"
                ),
            )
        )
        return c

    @staticmethod
    def plot_multi_pie_line(p):
        df = p.df[p.df['信息来源'].isin(p.source_dict.keys())]
        keys = list(p.source_dict.keys())
        c = Line()
        c.add_xaxis(keys)
        sentiments = ['情感正向', '情感中立', '情感负向']
        for s in sentiments:
            values = []
            for k in keys:
                values.append(len(df[(df['信息来源'] == k) & (df['情感倾向'] == s)]))
            c.add_yaxis(s, values)
        c.set_global_opts(title_opts=opts.TitleOpts(title='情感折线图'), xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-30)))
        return c