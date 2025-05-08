import pandas as pd

import sys
import csv
if __name__ == "__main__":

    log =  pd.read_csv('./data/Production_Data.csv',sep=',', encoding='utf-8')
    log = log.sort_values(by=['Case ID','Start Timestamp'])
    log = log.reset_index(drop=True)
    for i in range(0,len(log)):
        log.at[i,'Start Timestamp'] = log.at[i,'Start Timestamp'].split('.',2)[0]
        log.at[i,'Start Timestamp'] = pd.to_datetime(log.at[i,'Start Timestamp'],format='%Y/%m/%d %H:%M:%S')
    for i in range(0,len(log)):
        log.at[i,'Complete Timestamp'] = log.at[i,'Complete Timestamp'].split('.',2)[0]
        log.at[i,'Complete Timestamp'] = pd.to_datetime(log.at[i,'Complete Timestamp'],format='%Y/%m/%d %H:%M:%S')

    '''Log oszlop definíció és átnevezések'''
    column_dict = dict()
    column_dict['Case_id'] = 'Case ID' #eset azonosító
    column_dict['Start_timestamp'] = 'Start Timestamp' #esemény kezdete
    column_dict['End_timestamp'] = 'Complete Timestamp' #esemény vége
    column_dict['Event'] = 'Activity' #esemény
    column_dict['Resource'] = 'Resource' #eseményt végző erőforrás
    column_dict['Ordered'] = 'Work Order  Qty' #gyártandó mennyiség
    column_dict['Completed'] = 'Qty Completed' #elkészült mennyiség
    column_dict['Rejected'] = 'Qty Rejected' #visszautasított mennyiség
    column_dict['MRB'] = 'Qty for MRB' #MRB mennyiség
    column_dict['Part'] = 'Part Desc.' #alkatrész neve

    log.rename(columns={column_dict['Start_timestamp']:'start:timestamp',column_dict['End_timestamp']:'time:timestamp'
                        ,column_dict['Event']:'concept:name',column_dict['Case_id']:'case:concept:name',column_dict['Resource']:'org:resource'
                          ,column_dict['Ordered']:'Ordered',column_dict['Completed']:'Completed',column_dict['Rejected']:'Rejected',column_dict['MRB']:'MRB'
                          ,column_dict['Part']:'Part'},inplace=True)


    ##
    '''Célgép szűrés a log-on'''
    fmachine = 'Grinding Rework - Machine 27'  # célgép definiálás
    part = 'Cable Head'  # alkatrész definiálás
    flog = log[log['Part'] == part]
    flog = flog.reset_index(drop=True)
    fcase_ids = list(flog['case:concept:name'].unique())
    filt_idx = []
    for cid in fcase_ids:
        filt_log = flog[flog['case:concept:name'] == cid]
        idx = list(filt_log.index)
        machines = list(filt_log['concept:name'].unique())
        if fmachine in machines:
            filt_idx = filt_idx + idx

    fm_log = flog.iloc[filt_idx]
    fm_log.to_csv('./data/cable_head_mach_27.csv',index=False)

    '''Gyakori elemhalmaz keresés a célgépre szűrt log-ban'''
    operators = list(fm_log['Worker ID'].unique())
    operator_codes = []
    for i in range(1, 1 + len(operators)): operator_codes.append(i)
    oper_dict = dict(zip(operators, operator_codes))
    oper_dict_inv = dict(zip(operator_codes, operators))

    traces = [['@CONVERTED_FROM_TEXT']]
    for i in list(oper_dict_inv.keys()):
        text = '@ITEM=' + str(i) + '=' + oper_dict_inv[i]
        traces.append([text])
    tids = list(fm_log['case:concept:name'].unique())
    for tid in tids:
        slog = fm_log[fm_log['case:concept:name'] == tid]
        idx = list(slog.index.values)
        d = []
        for index in idx: d.append(oper_dict[slog.at[index, 'Worker ID']])
        d2 = ''
        for k in range(0, len(d)):
            d2 = d2 + str(d[k]) + ' '
        seq = [d2[:-1]]
        traces.append(seq)

    with open('./data/traces_csoft_oper.csv', 'w', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONE, delimiter='|', quotechar=',', escapechar='\\')
        writer.writerows(traces)
    ## filter log
    #log.to_csv('./data/pd_prepro.csv',index=False)
