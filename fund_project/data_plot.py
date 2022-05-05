import os
from pyecharts import options as opts
from pyecharts.charts import Pie
from pyecharts.charts import Bar
from pyecharts.charts import Line
from pyecharts.commons.utils import JsCode
from pyecharts.globals import ThemeType


class DataPlot:
    def __init__(self, file_path='./plot_res'):
        self.file_path = file_path
        if not os.path.exists(self.file_path):
            os.mkdir(self.file_path)

    def pie(self, name, value, fund_type, time_type, top_num):
        pic_name = f'{fund_type}基金涨跌幅'
        tips = f'[{fund_type}]基金{time_type}涨跌幅前{top_num}名'
        pie_path = os.path.join(self.file_path, 'fund_pie')
        if not os.path.exists(pie_path):
            os.mkdir(pie_path)
        c = (
            Pie()
                .add(
                "",
                [list(z) for z in zip(name, value)],
                # 饼图的中心（圆心）坐标，数组的第一项是横坐标，第二项是纵坐标
                # 默认设置成百分比，设置成百分比时第一项是相对于容器宽度，第二项是相对于容器高度
                center=["35%", "50%"],
            )
                .set_colors(["blue", "green", "yellow", "red", "pink", "orange", "purple"])  # 设置颜色
                .set_global_opts(
                title_opts=opts.TitleOpts(title="" + str(tips)),
                legend_opts=opts.LegendOpts(type_="scroll", pos_left="70%", orient="vertical"),  # 调整图例位置
            )
                .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}"))
                .render(os.path.join(pie_path, f'{pic_name}.html'))
        )

    def bars(self, name, dict_values):
        bar_path = os.path.join(self.file_path, 'fund_period_bar')
        if not os.path.exists(bar_path):
            os.mkdir(bar_path)
        # 链式调用
        c = (
            Bar(
                init_opts=opts.InitOpts(  # 初始配置项
                    theme=ThemeType.MACARONS,
                    animation_opts=opts.AnimationOpts(
                        animation_delay=1000, animation_easing="cubicOut"  # 初始动画延迟和缓动效果
                    ))
            )
                .add_xaxis(xaxis_data=name)  # x轴
                .add_yaxis(series_name="股票型", yaxis_data=dict_values['股票型'])  # y轴
                .add_yaxis(series_name="混合型", yaxis_data=dict_values['混合型'])  # y轴
                .add_yaxis(series_name="债券型", yaxis_data=dict_values['债券型'])  # y轴
                .add_yaxis(series_name="指数型", yaxis_data=dict_values['指数型'])  # y轴
                .add_yaxis(series_name="QDII型", yaxis_data=dict_values['QDII型'])  # y轴
                .set_global_opts(
                title_opts=opts.TitleOpts(title='涨跌幅', subtitle=' ',  # 标题配置和调整位置
                                          title_textstyle_opts=opts.TextStyleOpts(
                                              font_family='SimHei', font_size=25, font_weight='bold', color='red',
                                          ), pos_left="90%", pos_top="10",
                                          ),
                xaxis_opts=opts.AxisOpts(name='阶段', axislabel_opts=opts.LabelOpts(rotate=45)),
                # 设置x名称和Label rotate解决标签名字过长使用
                yaxis_opts=opts.AxisOpts(name='涨跌点'),

            )
                .render(os.path.join(bar_path, "基金各个阶段涨跌幅.html"))
        )

    def slider(self, name, value, fund_type, days):
        slider_path = os.path.join(self.file_path, 'fund_day_bar')
        if not os.path.exists(slider_path):
            os.mkdir(slider_path)
        tips = f'{fund_type}近{days}个交易日净值情况.html'
        c = (
            Bar(init_opts=opts.InitOpts(theme=ThemeType.DARK))
                .add_xaxis(xaxis_data=name)
                .add_yaxis(tips, yaxis_data=value)
                .set_global_opts(
                title_opts=opts.TitleOpts(title=tips.split('.')[0]),
                datazoom_opts=[opts.DataZoomOpts(), opts.DataZoomOpts(type_="inside")],
            )
                .render(os.path.join(slider_path, tips))
        )

    def line(self, dates, value_dict):
        kinds = ['股票型', '混合型', '债券型', '指数型', 'QDII型']
        line_path = os.path.join(self.file_path, 'fund_day_line')
        if not os.path.exists(line_path):
            os.mkdir(line_path)

        js_formatter = """function (params) {
                console.log(params);
                return '降水量  ' + params.value + (params.seriesData.length ? '：' + params.seriesData[0].data : '');
            }"""

        (
            Line(init_opts=opts.InitOpts(width="1600px", height="800px"))
                .add_xaxis(
                xaxis_data=dates
            )
                .add_yaxis(
                series_name=kinds[0],
                is_smooth=True,
                symbol="emptyCircle",
                is_symbol_show=False,
                # xaxis_index=1,
                color="#d14a61",
                y_axis=value_dict[kinds[0]],
                label_opts=opts.LabelOpts(is_show=False),
                linestyle_opts=opts.LineStyleOpts(width=2),
            )
                .add_yaxis(
                series_name=kinds[1],
                is_smooth=True,
                symbol="emptyCircle",
                is_symbol_show=False,
                color="#6e9ef1",
                y_axis=value_dict[kinds[1]],
                label_opts=opts.LabelOpts(is_show=False),
                linestyle_opts=opts.LineStyleOpts(width=2),
            )
                .add_yaxis(
                series_name=kinds[2],
                is_smooth=True,
                symbol="emptyCircle",
                is_symbol_show=False,
                color="#C7A0F2",
                y_axis=value_dict[kinds[2]],
                label_opts=opts.LabelOpts(is_show=False),
                linestyle_opts=opts.LineStyleOpts(width=2),
            )
                .add_yaxis(
                series_name=kinds[3],
                is_smooth=True,
                symbol="emptyCircle",
                is_symbol_show=False,
                color="#FFA983",
                y_axis=value_dict[kinds[3]],
                label_opts=opts.LabelOpts(is_show=False),
                linestyle_opts=opts.LineStyleOpts(width=2),
            )
                .add_yaxis(
                series_name=kinds[4],
                is_smooth=True,
                symbol="emptyCircle",
                is_symbol_show=False,
                color="#FFCF00",
                y_axis=value_dict[kinds[4]],
                label_opts=opts.LabelOpts(is_show=False),
                linestyle_opts=opts.LineStyleOpts(width=2),
            )
                .set_global_opts(
                legend_opts=opts.LegendOpts(),
                tooltip_opts=opts.TooltipOpts(trigger="none", axis_pointer_type="cross"),
                xaxis_opts=opts.AxisOpts(
                    type_="category",
                    axistick_opts=opts.AxisTickOpts(is_align_with_label=True),
                    axisline_opts=opts.AxisLineOpts(
                        is_on_zero=False, linestyle_opts=opts.LineStyleOpts(color="#d14a61")
                    ),
                    axispointer_opts=opts.AxisPointerOpts(
                        is_show=True, label=opts.LabelOpts(formatter=JsCode(js_formatter))
                    ),
                ),
                yaxis_opts=opts.AxisOpts(
                    type_="value",
                    splitline_opts=opts.SplitLineOpts(
                        is_show=True, linestyle_opts=opts.LineStyleOpts(opacity=1)
                    ),
                ),
            )
                .render(os.path.join(line_path, '基金净值走势.html'))
        )