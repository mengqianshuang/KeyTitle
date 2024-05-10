import argparse
import json, re
import random
import copy
import nltk
import pandas as pd
from tqdm.auto import tqdm
# replace miscellaneous information in body
def improve_body(origin_str, return_max_cnt = False): 
    maxcnt = 0 # placeholder number
    
    search = re.compile("```(?:.|\n)+?```") # ```codesnippt\ncode``` -> phofcode
    if return_max_cnt:
        maxcnt = max(maxcnt, len(search.findall(origin_str)))
    origin_str = search.sub(" phofcode ", origin_str)
    
    search = re.compile("\!\[(.*?)\]\(.+?\)") #![text](url) -> text phofimage
    if return_max_cnt:
        maxcnt = max(maxcnt, len(search.findall(origin_str)))
    origin_str = search.sub(lambda x: " " + x.group(1) + " phofimage ", origin_str)
    
    search = re.compile("(?<!\!)\[(.*?)\]\(.+?\)") #[text](url) -> text phofhyperlink
    if return_max_cnt:
        maxcnt = max(maxcnt, len(search.findall(origin_str)))
    origin_str = search.sub(lambda x: " " + x.group(1) + " phofhyperlink ", origin_str)
    
    search = re.compile("(https?|ftp)://[^\s/$.?#].[^\s]*") #url -> phofurl
    if return_max_cnt:
        maxcnt = max(maxcnt, len(search.findall(origin_str)))
    origin_str = search.sub(" phofurl ", origin_str)
    
    search = re.compile("- +\[ \].*") #line with unchecked tickbox in markdown (template provided useless desc) -> (remove)
    origin_str = search.sub("", origin_str)
    
    search = re.compile("(\*{1,})([^\s]+?)(\*{1,})") # avoid ***abc*** to be recognized as one token by nltk
    origin_str = search.sub(lambda x: " " + x.group(1) + " " + x.group(2) + " " + x.group(3) + " ", origin_str)
    
    search = re.compile("(\n\r)|(\r\n)|(\n)") # enter -> phofnewline
    if return_max_cnt:
        maxcnt = max(maxcnt, len(search.findall(origin_str)))
    origin_str = search.sub(" phofnewline ", origin_str)
    
    if return_max_cnt:
        return origin_str, maxcnt
    else:
        return origin_str, 0
    
# replace miscellaneous information in title
def improve_title(issue_title): 
    original_len = len(issue_title)
    issue_title = re.sub("^(\s*\[.*?\])+", "", issue_title) # remove starting [tag]
    pos = issue_title.find(": ")
    if -1 < pos < len(issue_title) - original_len / 2:
        issue_title = issue_title[pos + 1:].strip() # removing starting tag: 
    issue_title = re.sub("^(\s*\[.*?\])+", "", issue_title) # remove starting [tag] x2
    issue_title = re.sub("(\*{1,})(.+?)(\*{1,})", lambda x: x.group(2), issue_title) # remove emphasis
    return issue_title.strip()

# distinguish body hard to handle
def filter_body(issue_body, issue_body_tokenize):
    length = len(issue_body_tokenize)
    if length < 30:
        return True
    if length > 300:
        return True
    if len(re.findall("<[^<]+?>", issue_body)) > 0:
        return True
    return False

# TITLE expressiveness filter: title should with token number within [5,15] and contain no urls
def rule1checker(issue_title, issue_title_word):
    length = len(issue_title_word)
    if length < 5:
        return True
    if length > 15:
        return True
    if len(re.findall("(https?|ftp)://[^\s/$.?#].[^\s]*", issue_title)) > 0:
        return True
    return False

# TITLE relation filter: title hit token count should meet threshold
def rule2checker(issue_title_words, issue_body_words):
    body_words_set = set(issue_body_words)
    cnt_each = 0
    for word in issue_title_words:
        if word in body_words_set:
            cnt_each += 1
    if cnt_each <= len(issue_title_words) * 0.3: # less than 30% tokens hit, not one-sentence summarization
        return True
    return False
    
# TITLE copy (directly extract) filter: title copied substring length threshold
def rule3checker(issue_title, issue_title_tokenize, issue_body_tokenize):
    title_words = [x.lower() for x in issue_title_tokenize]
    body_words = [x.lower() for x in issue_body_tokenize]
    exp = ""
    # build substring location RE: (\s+AA)(\s+BB)(\s+CC)(\s+DD) to match AA BB CC (more postfix in title), BB CC DD (more prefix in title), AA BB DD (more tokens, such as punctuations, in title) etc. in body.
    for _ in title_words:
        _ = re.escape(_)
        exp += "(" + "\s+" + _ + ")?"
    re_iter = re.compile(exp)
    each_cnt = 0
    for s in re_iter.finditer(" " + " ".join(body_words)):
        each_each_cnt = 0
        for _ in s.groups():
            if _ is not None:
                each_each_cnt += 1
        each_cnt = max(each_cnt, each_each_cnt)
    if each_cnt >= len(title_words) * 0.7: # bad title - abnormal tag
        return True
    return False

def get_version_list(string):
    # e.g.  v1  V1.1  2.3  py3.6  1.2.3-alpha1  3.1rc  v2-beta3
    result = {}
    for item in re.findall("(?<=\W)((([vV][0-9]+)|([a-zA-Z_]*[0-9]+\w*(\.[a-zA-Z_]*[0-9]\w*)))([\.-]\w+)*)(?=\W)", string):#2,})", string):
        key = item[0].strip()
        if key not in result:
            result[key] = [len(result), 0] # order, term-freq
        result[key][1] += 1
    return result

def get_identifier_list(string):
    # e.g.  smallCamelCase  BigCamelCase  _underline_name_  test_123  func123Test
    result = {}
    for item in re.findall("(?<=\W)(([A-Z]*[a-z_][a-z0-9_]*)([A-Z_][a-z0-9_]*)+)(?=\W)", string):
        key = item[0].strip()
        if key not in result:
            result[key] = [len(result), 0] # order, term-freq
        result[key][1] += 1
    return result

def main():
    print ("loading raw json data... (this should take some minutes...)")
    all_issues1 = pd.read_csv("../data/valid_apache.csv")
    all_issues1.rename(columns={'summary':'title', 'text':'body'}, inplace=True)
    all_issues = all_issues1.to_json(orient="records", force_ascii=False)
    all_issues = eval(all_issues)
    print("原来文件长度：{}".format(len(all_issues)))
    # preprocessing procedure. - to filter out unhandlable samples and tailor unhandlable content from remained sampless
    print ("preprocessing...")
    preprocessed_issues = []
    invalid_issues = []
    for idx, issue in tqdm(enumerate(all_issues)):
        issue['body'], _ = improve_body(issue['body'])
        issue_body_tokenize = nltk.word_tokenize(issue['body'])
        if filter_body(issue['body'], issue_body_tokenize):
            invalid_issues.append(issue)
            continue
        issue['title'] = improve_title(issue['title'])
        preprocessed_issues.append(issue)
    print ("preprocess done. obtain", len(preprocessed_issues), "handlable issues.")
    
    # split dataset into 8:1:1 on all hanlable issues, 
    # to provide unified boundary and perform fair comparison between different refinement strategy
    sep1 = int(len(preprocessed_issues) * 0.8)
    sep2 = int(len(preprocessed_issues) * 0.9)

    print ("applying refinement rules...")

    valid_issues_train = []
    valid_issues_val = []
    valid_issues_test  = []

    for idx, issue in tqdm(enumerate(preprocessed_issues)):

        issue_body_tokenize = nltk.word_tokenize(issue['body'])
        issue_title_tokenize = nltk.word_tokenize(issue['title'])

        issue_title_words = [x.lower() for x in issue_title_tokenize if re.match("\S*[A-Za-z0-9]+\S*", x)]
        issue_body_words  = [x.lower() for x in issue_body_tokenize  if re.match("\S*[A-Za-z0-9]+\S*", x)] # lowercased words set

        if (rule1checker(issue['title'], issue_title_words)) \
         or(rule2checker(issue_title_words, issue_body_words)) \
         or(rule3checker(issue['title'], issue_title_tokenize, issue_body_tokenize)): # can disable any rule for ablation
            invalid_issues.append(issue)
            continue

        if idx < sep1:
            valid_issues_train.append(issue)
        elif idx < sep2:
            valid_issues_val.append(issue)
        else:
            valid_issues_test.append(issue)
            
    print ("refinement done. obtain", len(valid_issues_train), ",", len(valid_issues_val), ",", len(valid_issues_test), "issues for training, validation and testing.")
    
    
    # extract human-named tokens procedure
    print ("extracting human-named tokens...")

    valid_issues = valid_issues_train + valid_issues_val + valid_issues_test

    for idx, issue in enumerate(valid_issues):
        issue["_spctok"] = {}
        issue["_spctok"]["ver"] = get_version_list(" " + issue['body'] + " ")
        issue["_spctok"]["idt"] = get_identifier_list(" " + issue['body'] + " ")
    for idx, issue in enumerate(invalid_issues):
        issue["_spctok"] = {}
        issue["_spctok"]["ver"] = get_version_list(" " + issue['body'] + " ")
        issue["_spctok"]["idt"] = get_identifier_list(" " + issue['body'] + " ")
        
    # done. export :)
    save = [valid_issues_train, valid_issues_val, valid_issues_test]
    with open("refined_issues_ap.json", "w") as f:
        json.dump(save, f)
    with open("invalid_issues_ap.json", "w") as f:
        json.dump(invalid_issues, f)
    print("vaild issues: {}\ninvalid issues:{}".format(len(valid_issues), len(invalid_issues)))
    print ("preprocessing and refining success. refined sample set is saved to 'refined_" )
        
if __name__ == "__main__":
    main()