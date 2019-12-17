import json
import pandas as pd


# 1. build the rule_kcs_dict
def build_rule_kcs_dict():
    kcs_rule_dict = json.load(open("./data/kcs_rule_dict.json"))
    rule_kcs_dict = {}
    for key in kcs_rule_dict:
        rule_kcs_dict[kcs_rule_dict[key]] = key
    return rule_kcs_dict


# 2. build the kcs case dict
# def build_kcs_case_dict():
#     sql = """
#             SELECT kcs.resource_display_id__c, c.casenumber, c.subject, c.description, c.resolution1__c
#             FROM (SELECT  id, casenumber, subject, description, resolution1__c
#                   FROM stg_gss_case
#                   WHERE stg_curr_flg = true
#                       AND case_language__c='en'
#                       AND isdeleted = false
#                       AND ownerid != '00GA0000000XxxNMAS') c
#                 INNER JOIN stg_gss_case_rsrc_rltnshp kcs ON kcs.case__c = c.id
#             WHERE stg_curr_flg = true
#                 AND isdeleted = false
#                 AND type__c = 'Link'
#                 AND kcs.resource_display_id__c IN ({0})
#             ORDER BY kcs.resource_display_id__c
#     """.strip()
#
#     kcs_rule_dict = json.load(open("../data/kcs_rule_dict.json"))
#     kcs_string = ', '.join(kcs_rule_dict.keys())
#     kcs_cases_df = query_JDV(queryString=sql.format(kcs_string))
#     kcs_cases_df.to_pickle('../data/all_kcs_cases_df.pkl')
#     return kcs_cases_df.groupby(['resource_display_id__c'])
def load_kcs_case_dict():
    kcs_cases_df = pd.read_pickle('./data/all_kcs_cases_df.pkl')
    return kcs_cases_df.groupby(['resource_display_id__c'])


# 3. build case pairs
def build_case_pairs(rule_kcs_dict, kcs_case_group_result, customer_input, rule_list):
    kcs_list = []
    for rule in rule_list:
        try:
            kcs_list.append(rule_kcs_dict[rule])
        except:
            print("rule {0} is not in the database", rule)

    test_df = None
    for kcs in kcs_list:
        try:
            tmp_df = kcs_case_group_result.get_group(kcs)[['resource_display_id__c', 'casenumber', 'subject']]
            test_df = pd.concat([test_df, tmp_df])
        except:
            print("kcs {0} has no case linked", kcs)
    test_df.columns = ['caseb_kcs', 'caseb_id', 'caseb']
    test_df['casea_id'] = 'customer_input'
    test_df['casea'] = customer_input
    test_df['guid'] = test_df.index.values.tolist()
    test_df['label'] = 0
    return test_df
