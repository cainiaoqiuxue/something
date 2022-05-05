from spider import Spider
from data_plot import DataPlot


def solution2():
    dp = DataPlot()
    sp = Spider()
    for fund_type in sp.fund_type:
        time_type = '近一年'
        top_num = 10
        data = sp.get_fund_rank(fund_type, time_type, top_num)
        name = [d['fd_name'] for d in data]
        value = [d['yield'] for d in data]
        dp.pie(name, value, fund_type, time_type, top_num)


def solution3():
    dp = DataPlot()
    sp = Spider()
    fd_codes = dict()
    time_type = '近一周'
    for fund_type in sp.fund_type:
        fund_code = sp.get_fund_rank(fund_type, time_type, 10)[0]['fd_code']
        fd_codes[fund_type] = fund_code

    dict_values = dict()
    names = None
    for kind in fd_codes:
        data = sp.get_one_fund(fd_codes[kind])
        names = list(data.keys())
        dict_values[kind] = list(data.values())
    dp.bars(names, dict_values)


def solution4():
    dp = DataPlot()
    sp = Spider()
    fd_codes = dict()
    time_type = '近一周'
    days = 20
    for fund_type in sp.fund_type:
        fund_code = sp.get_fund_rank(fund_type, time_type, 10)[0]['fd_code']
        fd_codes[fund_type] = fund_code

    for kind in fd_codes:
        data = sp.get_one_fund_days(fd_codes[kind], days)
        name = [d['date'] for d in data]
        value = [d['nav'] for d in data]
        dp.slider(name, value, kind, days)


def solution5():
    dp = DataPlot()
    sp = Spider()
    fd_codes = dict()
    time_type = '近一周'
    days = 365
    for fund_type in sp.fund_type:
        fund_code = sp.get_fund_rank(fund_type, time_type, 10)[0]['fd_code']
        fd_codes[fund_type] = fund_code
    dates = None
    value_dict = dict()
    for kind in fd_codes:
        data = sp.get_one_fund_days(fd_codes[kind], days)
        dates = [d['date'] for d in data][::-1]
        value_dict[kind] = [d['value'] for d in data][::-1]
    dp.line(dates, value_dict)


if __name__ == '__main__':
    solution2()
    solution3()
    solution4()
    solution5()
