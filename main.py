import akshare as ak
import pandas as pd
import numpy as np
import time
import json
from typing import *
import pickle
from tqdm import tqdm
import heapq
from matplotlib import pyplot as plt

def test():
    stock_info_a_code_name_df = ak.stock_info_a_code_name().values
    all = {}
    for x in stock_info_a_code_name_df:
        all[x[0]] = x[1]
    save_obj(all, 'allAcode')
    print(len(all), all)


def get_interest(A, B, Acodes, output):
    A = load_obj(A)
    B = load_obj(B)
    Acodes = load_obj(Acodes)
    C = {}
    for key in A:
        if key in B: continue
        if key not in Acodes: continue
        C[key] = A[key]
    save_obj(C, output)
    return C


def get_sz_name_code():
    stock_info_sz_name_code = ak.stock_info_sz_name_code()
    print(stock_info_sz_name_code)
    stock_info_sz_name_code.to_excel("深交所所有股票.xlsx", index=False)
    all = {}
    stock_info_sz_name_code = stock_info_sz_name_code.values
    for i in range(len(stock_info_sz_name_code)):
        code, name = stock_info_sz_name_code[i][1], stock_info_sz_name_code[i][2]
        type = stock_info_sz_name_code[i][0]
        date = stock_info_sz_name_code[i][3]
        if type == "科创板" and date > "20200612":
            continue
        all[code] = name
    print(all)
    save_obj(all, "Allcode2name_sz")


def get_sh_name_code():
    stock_info_sh_name_code = ak.stock_info_sh_name_code()
    print(stock_info_sh_name_code)
    stock_info_sh_name_code.to_excel("上交所所有股票.xlsx", index=False)
    all = {}
    stock_info_sh_name_code = stock_info_sh_name_code.values
    for i in range(len(stock_info_sh_name_code)):
        code, name = stock_info_sh_name_code[i][0], stock_info_sh_name_code[i][1]
        all[code] = name
    print(all)
    save_obj(all, "Allcode2name_sh")


def save_obj(obj, name):
    with open('' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def getexpand(file):
    sz, sh_new, sh_old = list(pd.read_excel(file, sheet_name=None))
    df_sz = pd.read_excel(file, sheet_name=sz).values
    df_sh_new = pd.read_excel(file, sheet_name=sh_new).values
    df_sh_old = pd.read_excel(file, sheet_name=sh_old).values
    expand_sh, expand_sz = {}, {}
    for i in range(len(df_sz)):
        if df_sz[i][3] and df_sz[i][3] == "新增":
            code = str(df_sz[i][1])
            newcode = '0' * (6 - len(code)) + str(code)
            expand_sz[newcode] = str(df_sz[i][2])
    print("SZ:\n", len(expand_sz), expand_sz)
    tmp = {}
    for i in range(len(df_sh_old)):
        code, name = str(df_sh_old[i][1]), str(df_sh_old[i][2])
        tmp[code] = name
    for i in range(len(df_sh_new)):
        code, name = str(df_sh_new[i][1]), str(df_sh_new[i][2])
        if code not in tmp:
            print(code, name)
            newcode = '0' * (6 - len(str(code))) + str(code)
            expand_sh[newcode] = str(name)
    print("SH:\n", len(expand_sh), expand_sh)
    save_obj(expand_sz, 'expand_sz')
    save_obj(expand_sh, 'expand_sh')
    return expand_sz, expand_sh


def getexist(file):
    sz, sh_new, sh_old = list(pd.read_excel(file, sheet_name=None))
    df_sz = pd.read_excel(file, sheet_name=sz).values
    df_sh_new = pd.read_excel(file, sheet_name=sh_new).values
    df_sh_old = pd.read_excel(file, sheet_name=sh_old).values
    exist_sh, exist_sz = {}, {}
    for i in range(len(df_sz)):
        if df_sz[i][3] and df_sz[i][3] == "新增":
            continue
        code = str(df_sz[i][1])
        name = str(df_sz[i][2])
        newcode = '0' * (6 - len(code)) + code
        exist_sz[newcode] = name
    print("SZ:\n", exist_sz)
    exist_sh = {}
    for i in range(len(df_sh_old)):
        code, name = str(df_sh_old[i][1]), str(df_sh_old[i][2])
        newcode = '0' * (6 - len(code)) + code
        exist_sh[newcode] = name
    print("SH:\n", exist_sh)
    save_obj(exist_sh, "exist_sh")
    save_obj(exist_sz, "exist_sz")
    return exist_sh, exist_sz


def gethistory_sh(codes, start="20190819", end="20221024"):
    laststart = start
    res = {}
    for code, name in tqdm(codes.items()):
        code = str(code)
        code = '0' * (6 - len(code)) + code
        # stock_zh_a_hist_163_df = ak.stock_zh_a_hist_163(symbol="sh" + code, start_date="20210101", end_date="20220101")
        # stock_zh_a_hist_163_df = ak.stock_zh_a_daily(symbol="sh" + code, start_date=start, end_date=end,
        #                                              adjust="hfq").values
        stock_zh_a_hist_163_df = ak.stock_zh_a_daily(symbol="sh" + code, start_date=start, end_date=end).values
        try:
            res[code] = extract(code, name, stock_zh_a_hist_163_df)
        except Exception as e:
            print(code, len(stock_zh_a_hist_163_df), e)
            continue
        if laststart < stock_zh_a_hist_163_df[0][0].strftime("%Y%m%d"):
            laststart = stock_zh_a_hist_163_df[0][0].strftime("%Y%m%d")
    save_obj(res, 'History_interets_sh')
    return res


def gethistory_sz(codes, start="20190819", end="20221024"):
    # get mindate
    laststart = start
    res = {}
    for code, name in tqdm(codes.items()):
        code = str(code)
        code = '0' * (6 - len(code)) + code
        # stock_zh_a_hist_163_df = ak.stock_zh_a_hist_163(symbol="sz" + code, start_date="20210101", end_date="20220101")
        # stock_zh_a_hist_163_df = ak.stock_zh_a_daily(symbol="sz" + code, start_date=start, end_date=end,
        #                                              adjust="hfq").values
        stock_zh_a_hist_163_df = ak.stock_zh_a_daily(symbol="sz" + code, start_date=start, end_date=end).values
        try:
            res[code] = extract(code, name, stock_zh_a_hist_163_df)
        except Exception as e:
            print(code, len(stock_zh_a_hist_163_df), e)
            continue
        if laststart < stock_zh_a_hist_163_df[0][0].strftime("%Y%m%d"):
            laststart = stock_zh_a_hist_163_df[0][0].strftime("%Y%m%d")
    save_obj(res, 'History_interets_sz')
    return res


def extract(code, name, values):
    startdate = values[0][0]
    date = values[:, 0]
    liutong = []
    chengjiao = []
    for i in range(len(values)):
        liutong.append(values[i][6] * values[i][4])
        chengjiao.append(values[i][5] * values[i][4])

    res = {"code": code, "name": name, "start": startdate, "date": date, "liutong": liutong, "chengjiao": chengjiao}
    return res


def get_sh_all(dates):
    res = {}
    for i in tqdm(range(len(dates))):
        # time.sleep(0.5)
        # date = dates[i].strftime("%Y%m%d")
        date = dates[i]
        try:
            df = ak.stock_sse_deal_daily(date)
        except:
            print(dates[i])
            continue

        lt_row_index = df[df["单日情况"] == "流通市值"].index.tolist()[0]
        cj_row_index = df[df["单日情况"] == "成交金额"].index.tolist()[0]
        liutong_price = df["主板A"][lt_row_index] * 100000000
        chengjiao_price = df["主板A"][cj_row_index] * 100000000

        res[date] = {"date": date, "liutong": liutong_price, "chengjiao": chengjiao_price}
    save_obj(res, "shAall")
    return res


def get_sz_all(dates):
    '''
    返回总和，计算需要计算差值
    :return:
    '''
    res = {}
    ori_lt, ori_cj = 0, 0
    for i in tqdm(range(len(dates))):
        # time.sleep(0.5)
        # date = dates[i].strftime("%Y%m%d")
        date = dates[i]
        try:
            df = ak.stock_szse_summary(date)
        except:
            print(dates[i])
            continue
        zhuA_index = df[df["证券类别"] == "主板A股"].index.tolist()[0]
        chuangyeA_index = df[df["证券类别"] == "创业板A股"].index.tolist()[0]
        # zxA_index = df[df["证券类别"] == "中小板"].index.tolist()[0]
        # all_index = df[df["证券类别"] == "股票"].index.tolist()[0]
        # B_index = df[df["证券类别"] == "主板B股"].index.tolist()[0]
        # liutong_price = df["流通市值"][all_index] - df["流通市值"][B_index]
        # chengjiao_price = df["成交金额"][all_index] - df["成交金额"][B_index]

        liutong_price = df["流通市值"][zhuA_index] + df["流通市值"][chuangyeA_index]
        chengjiao_price = df["成交金额"][zhuA_index] + df["成交金额"][chuangyeA_index]

        # ori_lt, ori_cj = df[0][4], df[0][2]
        # if i==0:continue
        res[date] = {"date": date, "liutong": liutong_price, "chengjiao": chengjiao_price}
    save_obj(res, "szAall")
    return res


def export(input, output):
    df = pd.DataFrame(input)
    df.to_excel(output, index=False)


def getdates(file):
    firstdate = "20221024"
    values = load_obj(file)
    res = []
    for key, value in values.items():
        tmpdate = value["start"].strftime("%Y%m%d")
        if tmpdate < firstdate:
            firstdate = tmpdate
            res = value["date"]
    return res


def process(dictfile, output):
    history = load_obj(dictfile)
    res = {}
    prefix = {}
    for key in history:
        prefix[key] = {"lt": 0, "cj": 0}
    for code in history:
        if code not in res:
            res[code] = {"name": history[code]["name"]}
        for i in range(len(history[code]['date'])):
            day = history[code]['date'][i].strftime("%Y%m%d")
            lt = history[code]['liutong'][i]
            cj = history[code]['chengjiao'][i]
            prefix[code]["lt"] += lt
            prefix[code]["cj"] += cj
            res[code][day] = {"prefixlt": prefix[code]["lt"], "prefixcj": prefix[code]["cj"],
                              "index": i + 1}
    save_obj(res, output)
    return res


def processA(dictfile, output):
    history = load_obj(dictfile)
    print(history)
    res = {}
    prefix = {"lt": 0, "cj": 0}
    i = 0
    for date in history:
        # lt = history[date]['liutong'] * 100000000
        # cj = history[date]['chengjiao'] * 100000000
        lt = history[date]['liutong']
        cj = history[date]['chengjiao']
        if cj == 0:
            print(date)
        prefix["lt"] += lt
        prefix["cj"] += cj
        res[date] = {"prefixlt": prefix["lt"], "prefixcj": prefix["cj"],
                     "index": i + 1}
        i += 1
    save_obj(res, output)
    return res


def calacc(reslist, expand):
    num = 0
    for i in range(len(expand)):
        score, code = heapq.heappop(reslist)
        if code in expand:
            num += 1
    return num / len(expand)


def gridsearch(allA, interest, expand, output):
    allAcodes = load_obj('allAcode')
    allA = load_obj(allA)
    interest = load_obj(interest)
    expand = load_obj(expand)
    alldates = load_obj("all_dates")
    res = {}
    for i in tqdm(range(len(alldates))):
        preday = alldates[i]
        if preday < '20220101': continue
        if preday not in allA: continue
        preA = allA[preday]
        preAlt, preAcj, preAidx = preA['prefixlt'], preA['prefixcj'], preA['index']
        for j in range(i + 1, len(alldates)):
            postdate = alldates[j]
            if postdate < '20220606': continue
            if postdate not in allA: continue
            postA = allA[postdate]
            postAlt, postAcj, postAidx = postA['prefixlt'], postA['prefixcj'], postA['index']
            meanAlt, meanAcj = (postAlt - preAlt) / (postAidx - preAidx), (postAcj - preAcj) / (postAidx - preAidx)
            queue = []
            heapq.heapify(queue)
            for code in interest:
                if code not in allAcodes:
                    continue
                value = interest[code]
                sortkeys = list(value.keys())[1:]
                # sortkeys.sort()
                firstday, lastday = sortkeys[0], sortkeys[-1]
                if preday not in value:
                    prelt, precj, preidx = 0, 0, 0
                else:
                    prelt, precj, preidx = value[preday]["prefixlt"], value[preday]["prefixcj"], value[preday]["index"]

                if postdate in value:
                    postlt, postcj, postidx = value[postdate]["prefixlt"], value[postdate]["prefixcj"], value[postdate][
                        "index"]
                elif lastday < postdate:
                    postlt, postcj, postidx = value[lastday]["prefixlt"], value[lastday]["prefixcj"], value[lastday][
                        "index"]
                elif firstday > postdate:
                    postlt, postcj, postidx = 0, 0, preidx + 1
                else:
                    tmpdate = firstday
                    for x in sortkeys:
                        if x > postdate:
                            break
                        else:
                            tmpdate = x
                    postlt, postcj, postidx = value[tmpdate]["prefixlt"], value[tmpdate]["prefixcj"], value[tmpdate][
                        "index"]
                try:
                    meanlt, meancj = (postlt - prelt) / (max(postidx - preidx, 1)), (postcj - precj) / (
                        max(postidx - preidx, 1))
                    score = 2 * (meanlt / meanAlt) + (meancj / meanAcj)
                except Exception as e:
                    print(meanAlt, meanAcj, prelt, precj, preidx, postlt, postcj, postidx)
                    print(e)
                    continue
                heapq.heappush(queue, [-1 * score, code])
            if len(queue) > len(expand):
                acc = calacc(queue, expand)
                res['-'.join([preday, postdate])] = acc

    # print(res)
    save_obj(res, output)


def getmaxvalue(resultfile):
    res = load_obj(resultfile)
    maxacc = 0
    finaldate = ""
    for x in res:
        if res[x] > maxacc:
            maxacc = res[x]
            finaldate = x

    print(resultfile, maxacc, finaldate)


def getallA(start="20190819", end="20221024"):
    codes = load_obj('allAcode')
    alldates = load_obj('all_dates')
    res_sh = {}
    res_sz = {}
    for code, name in tqdm(codes.items()):
        code = str(code)
        code = '0' * (6 - len(code)) + code
        stock_zh_a_hist_163_df = None
        flag = "sh"
        try:
            stock_zh_a_hist_163_df = ak.stock_zh_a_daily(symbol="sh" + code, start_date=start, end_date=end).values
        except:
            flag = "sz"
            pass
        try:
            if not stock_zh_a_hist_163_df or len(stock_zh_a_hist_163_df) == 0:
                stock_zh_a_hist_163_df = ak.stock_zh_a_daily(symbol="sz" + code, start_date=start, end_date=end).values
        except:
            continue
        try:
            tmp = extract(code, name, stock_zh_a_hist_163_df)
        except Exception as e:
            print(code, len(stock_zh_a_hist_163_df), e)
            continue
        date = tmp['date']
        liutong = tmp['liutong']
        chengjiao = tmp['chengjiao']
        if not len(date) == len(liutong) == len(chengjiao):
            print(code, name, len(date), len(liutong), len(chengjiao))
            continue
        for i in range(len(date)):
            if flag == "sh":
                if date[i] not in res_sh:
                    res_sh[date[i]] = {'liutong': 0, 'chengjiao': 0}
                res_sh[date[i]]['liutong'] += liutong[i]
                res_sh[date[i]]['chengjiao'] += chengjiao[i]
            else:
                if date[i] not in res_sz:
                    res_sz[date[i]] = {'liutong': 0, 'chengjiao': 0}
                res_sz[date[i]]['liutong'] += liutong[i]
                res_sz[date[i]]['chengjiao'] += chengjiao[i]
    save_obj(res_sh, 'A_sh_History')
    save_obj(res_sz, 'A_sz_History')
    print("SH:\n", len(res_sh), res_sh)
    print("SZ:\n", len(res_sz), res_sz)


if __name__ == '__main__':
    # 距离上次扩容：2019年8月19日～2022年10月24日; 20220606
    # get_sz_name_code()
    # get_interest('Allcode2name_sz', 'exist_sz', 'allAcode', 'code2name_wo_existsz')

    # get_sz_all(load_obj("all_dates"))
    # processA('szAall', "Mean_AAll_sz")

    # get_sh_all(load_obj("all_dates"))
    # processA('shAall', "Mean_AAll_sh")

    # getallA()
    # print(load_obj('A_sh_History'))
    # print(load_obj("Mean_AAll_sh"))

    # print(load_obj('szAall'))
    # processA('A_sh_History', "Mean_AAll_sh")
    # processA('A_sz_History', "Mean_AAll_sz")

    # gridsearch(allA="Mean_AAll_sh", interest="Mean_interets_sh", expand="expand_sh")
    # getexpand('expand.xlsx')
    # get_interest('Allcode2name_sh', 'exist_sh', 'allAcode', 'code2name_wo_existsh')
    # get_interest('Allcode2name_sz', 'exist_sz', 'allAcode', 'code2name_wo_existsz')
    # codes = load_obj('code2name_wo_existsh')
    # gethistory_sh(codes)
    # codes = load_obj('code2name_wo_existsz')
    # gethistory_sz(codes)
    # process('History_interets_sh', 'Mean_interets_sh')
    # process('History_interets_sz', 'Mean_interets_sz')

    # gridsearch(allA="Mean_AAll_sh", interest="Mean_interets_sh", expand="expand_sh", output='Result_sh')
    # gridsearch(allA="Mean_AAll_sz", interest="Mean_interets_sz", expand="expand_sz", output='Result_sz')
    # getmaxvalue('Result_sh')
    # getmaxvalue('Result_sz')

    res = load_obj("Result_sz")
    X,Y = [],[]

    for x in res:
        X.append(x)
        Y.append(res[x])
    plt.plot(X,Y)
    # 展示图形
    plt.show()
