import fitz
import Levenshtein
import re
type_dict  = {
    "FULL_UPPER": 1,
    "Nomal": 0,
    "Upper each word": 2,
    "unknown": 3
}
bullet_point_patterns = r"^(\u2022|\u2023|\u25B6|\u25C0|\u25E6|\u25A0|\u25A1|\u25AA|\u25AB|\u2013)"
bullet_point_patterns2 = r"^(\u2022|\u2023|\u25B6|\u25C0|\u25E6|\u25A0|\u25A1|\u25AA|\u25AB|\u2013)$"
def check_bullet_point(text):
    if re.search(bullet_point_patterns, text):
        return True
    return False
def check_bullet_point2(text):
    if re.search(bullet_point_patterns2, text):
        return True
    return False
unnessary_words = [".","of","a","an","the","in","on","at","to","and","for","about","with","by","from","as","or","into","over","like","through","under","between","after","before","during","since","until","against","towards","upon","within","without","along","across","behind","beyond","except","inside","outside"]
heading_pattern = r"^(Chapter\s|Part\s|Section\s)?[0-9IVXLCDM]{1,3}(\.|\:|\,|\s)([1-9IVXLCDM]{1,3}(\.|\:|\,)?)*\s"
def remove_chapter_marking(text):
    text = re.sub(heading_pattern, "", text, re.I).strip()
    return text
def check_captions(text):
    if re.search(r"^(Figure|Table|Listing)\s[A-Z0-9IVXLCDM]{1,5}\-[A-Z1-9IVXLCDM]{1,4}", text.strip()):
        return True

heading_pattern = r"^[0-9]{1,2}\.[0-9]{1,2}\s[A-Z]"
def is_heading_manual_level_2(text):
    if text.strip().startswith("1.1"):
        print()
    if re.search(heading_pattern, text):
        return True
    return False
def check_headings(text, full_bookmarks, allow_level = [1], page_numer = 0, previous = {}, line_type = 0):
    if text.strip() in [""]:
        return None
    for b in full_bookmarks:
        # key = b[1]+"_"+ str(b[2])
        # if key in used_bookmarks:
        #     continue
        if b[0] in allow_level and b[2] == page_numer:# this text is a heading with a valid level
            if Levenshtein.ratio(text.strip(), b[1]) > 0.8:
                if not (b[1][-1] not in ["!", "?", ".",":"] and text[-1] in ["!", "?", ".", ":"]):
                    return b
            text = remove_chapter_marking(text)
            text2 = remove_chapter_marking(b[1])
            if line_type == 1:
                text = text.upper()
                text2 = text2.upper()
            if Levenshtein.ratio(text.strip(), text2.strip()) > 0.9:
                if not (b[1][-1] not in ["!", "?", ".",":"] and text[-1] in ["!", "?", ".", ":"]):
                    return b
               
            # if len(text.split(" ")) >= 3 and text in text2:
            #     return b
            if len(previous) > 0:
                if previous["is_heading"] == b[0] and previous["page"] == b[2]:
                    if text in text2 and previous["text"][-1] not in ["!", "?", "."] and len(text) >3:
                        return b
    if is_heading_manual_level_2(text):
        return [2, text, page_numer]        
    return None

def get_full_bookmarks(pdf):
    with fitz.open(pdf) as file_temp:
        bookmarks = file_temp.get_toc()
        for b in bookmarks:
            b[1] = fix_text(b[1])
    return bookmarks
def get_important_bookmarks(pdf, level = [1]): #try to detect important headling for split
    return_list = []
    current_level = 1
    with fitz.open(pdf) as file_temp:
            bookmarks = file_temp.get_toc()
            for b in bookmarks:
                if b[0] in level:
                    b[1] = fix_text(b[1])
                    return_list.append(b)
    if len(return_list) < 2:
        current_level = 2
        return_list = get_important_bookmarks(pdf, level = [1,2])
    return return_list, current_level
def check_heading1(text,book_marks, page_number = 0, relax = False, used_bookmarks = {}):
    for b in book_marks:
        key = b[1]+"_"+ str(b[2])
        if key in used_bookmarks: # only allow each heading to be used once in same page
            continue
        if b[2] != page_number:
            continue
        if Levenshtein.ratio(text.strip(), b[1]) > 0.8 and b[2] == page_number:
            if not( b[1][-1] not in ["!", "?", ".",":"] and text[-1] in ["!", "?", ".", ":"]):
                return b
        
        text = remove_chapter_marking(text)
        text2 = remove_chapter_marking(b[1])
        if Levenshtein.ratio(text.strip(), text2.strip()) > 0.8 and b[2] == page_number:
            return b
        if len(text.split(" ")) >= 3 and text in text2:
                return b
        if relax:
            if Levenshtein.ratio(text.strip(), text2.strip().replace(" ","")) > 0.8 and b[2] == page_number:
                return b
    return None

def get_type(text):
    if text.strip() in [""]:
        return type_dict["Nomal"]
    text = re.sub(r"(\!|\.|\?|\:)", "", text).strip()
    splits = text.strip().split(" ")
    if splits[0].isalpha() and splits[0].islower() and len(splits[0]) > 1:
        return type_dict["Nomal"]
    splits = [split for split in splits if (split not in unnessary_words) and (split.isalnum())]
    if len(splits) == 0:
        return type_dict["Nomal"]
    if len(splits) == 1:
        if splits[0].isupper():
            return type_dict["FULL_UPPER"]
        if splits[0][0].isupper() and splits[0][1:].islower():
            return type_dict["Upper each word"]
        return type_dict["Nomal"]
    count_full_upper = 0
    count_upper_each_word = 0
    total = 0
    count_unknown = 0

    for split in splits:
        if split.isnumeric() or ( not split.isalnum() and len(split) < 2):
            continue
        total += 1
        if split.islower() and not split.isnumeric():
            return type_dict["Nomal"]
        if split[0].isupper() and split[1:].islower():
            count_upper_each_word += 1
        if split.isupper():
            count_full_upper += 1
        upper = 0
        lower = 0
        for char in split:
            if char.isupper():
                upper += 1
            if char.islower():
                lower += 1
        if upper > 1 and lower > 0:
            count_unknown += 1
    if total == 0:
        return type_dict["Nomal"]
    # if count_full_upper/total >= 0.8:
    #     return type_dict["FULL_UPPER"]
    if count_upper_each_word/total > 0.8:
        return type_dict["Upper each word"]
    # if count_upper_each_word/total > 0.4 and check_headings(text.strip()):
    #     return type_dict["Upper each word"]
    if count_unknown/total > 0.4:
        return type_dict["Upper each word"]
    count = 0
    total = 0 
    for c in text.strip():
        if not c.isalpha():
            continue
        total += 1
        if c.isupper():
            count += 1
    if count/total > 0.8:
        return type_dict["FULL_UPPER"]
    return type_dict["Nomal"]
def is_removable(text):
    raw_text = text

    if text.strip() in [""]:
        return True
    count = 0
    for c in text:
        if c.isalnum():
            count += 1
    if count/len(raw_text) < 0.2:
        return True
    
    text = re.sub(r"(\!|\.|\?|\:|\;|\"|\+|\=|\_|\\|\/|\(|\)|\{|\}|\-|\,|\>|\<|\]|\[|\@|\#|\$|\%|\^|\*)", " ", text).strip()
    text = text.replace("  ", " ")
    splits = text.split(" ")
    splits = [split for split in splits if split.lower() not in ["a", "an","of","in","on","at","to"]]
    count = 0
    total = 0
    for s in splits:
        
        if s.isalpha():
            total += 1
        if s.isalpha() and len(s) < 2:
            count += 1
    if total == 0:
        return False
    if count/total > 0.5:
        return True
    return False
def get_text(pdf, full_bookmarks = None, level1_bookmarks = None, starting_page = 1, current_level = 1):
    recording = {}
    analyzed_text = []
    my_pages = []
    with fitz.open(pdf) as doc:
        count = 1
        used_bookmarks = {}
        used_bookmarks1 = {}
        for page in doc:
            if count < starting_page:
                count += 1
                continue
            each_page = []
            # a = page.mediabox
            text =  page.get_text()
            text = fix_text(text)
            lines = text.split("\n")
            # if count == 21:
            #     print()
            previous = {}

            for i in range(len(lines)):

                line = lines[i].strip()
                if  re.search(r"^([\[\-\s]{0,5})[0-9IVX]{1,4}([\]\-\s]{0,5})$", line, re.IGNORECASE) and (i <= 3 or i >= len(lines) - 3):
                    continue #remove page number
                type = get_type(line)
                # if is_gibberish(line):
                #     print(line)
                #     continue
                # heading1 = check_heading1(line, level1_bookmarks, page_number = count)
                # if heading1 != None:
                    
                #     key1 = line+"_"+ str(heading1[2])#text_page
                #     if key1 in used_bookmarks1:
                #         is_breaker = False
                #     else:
                #         is_breaker = True
                #         used_bookmarks1[key1] = 1
                # else:
                #     is_breaker = False
                is_breaker = False
                
                bookmark = check_headings(line, full_bookmarks, allow_level = [1,2,3,4], page_numer = count, previous= previous, line_type = type)
                if bookmark != None:
                    
                    key = line+"_"+ str(bookmark[2])
                    if key in used_bookmarks:
                        is_heading = -1
                    else:
                        is_heading = bookmark[0]
                        if is_heading <= current_level and is_heading != -1:
                            is_breaker = True
                        used_bookmarks[key] = 1
                        previous = {}
                        previous["text"] = line
                        previous["page"] = count
                        previous["type"] = type
                        previous["is_breaker"] = is_breaker
                        previous["is_heading"] = is_heading
                else:
                    is_heading = -1
                    previous = {}
                each_page.append({"text":line, "type":type, "is_breaker":is_breaker, "is_heading":is_heading, "page_number":count, "size": 10, "order": i, "page": count, "total_lines": len(lines)})
            for i in range(len(lines)):
                line = lines[i].strip()
                if i <= 3 or i >= len(lines) - 3:
                    if line not in recording:
                        recording[line] = 1
                    else:
                        recording[line] += 1
                    if re.search(r"^[0-9]{1,6}\s", line) or re.search(r"\s[0-9]{1,6}$", line):
                        removed_page_number = re.sub(r"(^[0-9]{1,6}\s|\s[0-9]{1,6}$)", "", line).strip()
                        if removed_page_number not in recording:
                            recording[removed_page_number] = 1
                        else:
                            recording[removed_page_number] += 1
            count += 1
            my_pages.append(each_page)
        recording = {k: v for k, v in recording.items() if v > 1}
    return_data = []
    #remove header and footer
    for page in my_pages:
        for a in page:
            line = a["text"].strip()
            order = a["order"]
            if order > 3 and order < len(page) - 3:
                return_data.append(a)
                continue
            if line in recording and a["is_heading"] == -1 and a["is_breaker"] == False:
                continue
            temp_text = re.sub(r"(^[0-9]{1,6}\s|\s[0-9]{1,6}$)", "", line).strip()
            if temp_text in recording and a["is_heading"] == -1 and a["is_breaker"] == False:
                continue
            if is_removable(line) and a["is_heading"] == -1 and a["is_breaker"] == False:
                continue
            return_data.append(a)
    return return_data




def re_structure(parts: list):
    sections = []
    start = parts[0]["text"]
    current_size = parts[0]["size"]
    current_type = parts[0]["type"]
    middle = " "
    hyphen_flag = False
    previous_break = False
    enumeration = False
    is_breaker = False
    previous_p = parts[0]
    for p in parts[1:]:
        text = p["text"]
        size = p["size"]
        type = p["type"]
        if text.strip() in [""]:
            continue
        if start.strip() == "":
            start = text
            middle = " "
            current_size = size
            hyphen_flag = False
            previous_break = False
            continue
        if start.strip()[-1] == "-":
            middle = ""
            start = start.strip()[:-1]#remove hyphen
            hyphen_flag = True
        if text.strip() == "-":
            hyphen_flag = True
            middle = ""
            continue
        if hyphen_flag:
            if text.strip()[0].isalpha() and text.strip()[0].islower():
                start += middle + text
                middle = " "
                hyphen_flag = False
                continue
        if start.strip()[-1] in ["!", "?"]:
            sections.append({"text": start.replace("  ", " "),"type": current_type, "size": current_size, "is_breaker": previous_p["is_breaker"], "is_heading": previous_p["is_heading"], "page": previous_p["page"], "total_lines": previous_p["total_lines"]})
            start = text
            previous_p = p
            middle = " "
            current_size = size
            hyphen_flag = False
            previous_break = False
            current_type = type
            continue
        if re.search(r"^See\s", text.strip()):
            sections.append({"text": start.replace("  ", " "),"type": current_type, "size": current_size, "is_breaker": previous_p["is_breaker"], "is_heading": previous_p["is_heading"], "page": previous_p["page"], "total_lines": previous_p["total_lines"]})
            start = text
            previous_p = p
            middle = " "
            current_size = size
            hyphen_flag = False
            previous_break = False
            current_type = type
            continue
        if re.search(r"^See\s", start.strip()) and start.endswith("."):
            sections.append({"text": start.replace("  ", " "),"type": current_type, "size": current_size, "is_breaker": previous_p["is_breaker"], "is_heading": previous_p["is_heading"], "page": previous_p["page"], "total_lines": previous_p["total_lines"]})
            start = text
            previous_p = p
            middle = " "
            current_size = size
            hyphen_flag = False
            previous_break = False
            current_type = type
            continue
        if re.search(r"^[0-9]{1,2}\s[A-Za-z]{3,4}", text) and p['is_heading'] == -1:
            sections.append({"text": start.replace("  ", " "),"type": current_type, "size": current_size, "is_breaker": previous_p["is_breaker"], "is_heading": previous_p["is_heading"], "page": previous_p["page"], "total_lines": previous_p["total_lines"]})
            start = text
            previous_p = p
            middle = " "
            current_size = size
            hyphen_flag = False
            previous_break = False
            current_type = type
            continue
        if re.search(r"^[0-9]{1,2}\s[A-Za-z]{3,4}", start) and previous_p['page'] != p['page']:
            sections.append({"text": start.replace("  ", " "),"type": current_type, "size": current_size, "is_breaker": previous_p["is_breaker"], "is_heading": previous_p["is_heading"], "page": previous_p["page"], "total_lines": previous_p["total_lines"]})
            start = text
            previous_p = p
            middle = " "
            current_size = size
            hyphen_flag = False
            previous_break = False
            current_type = type
            continue
        if check_bullet_point2(start.strip()):
            start += " " + text
            # print(start)
            previous_p = p
            middle = " "
            current_size = size
            hyphen_flag = False
            previous_break = False
            current_type = type
            continue
        if check_bullet_point(text.strip()):
            sections.append({"text": start.replace("  ", " "),"type": current_type, "size": current_size, "is_breaker": previous_p["is_breaker"], "is_heading": previous_p["is_heading"], "page": previous_p["page"], "total_lines": previous_p["total_lines"]})
            start = text
            previous_p = p
            middle = " "
            current_size = size
            hyphen_flag = False
            previous_break = False
            current_type = type
            continue
        if (previous_p["is_breaker"] and p["is_breaker"] and start.strip()[-1] not in ["!", "?", "."] and previous_p["page"] == p["page"]) or (previous_p["is_heading"]== p["is_heading"] and p["is_heading"] != -1 and start.strip()[-1] not in ["!", "?", "."] and previous_p["page"] == p["page"]):
            start += middle + text
            middle = " "
            continue
        if previous_p["is_breaker"] or previous_p["is_breaker"]!= p["is_breaker"] or previous_p["is_heading"]!= p["is_heading"]:
            sections.append({"text": start.replace("  ", " "),"type": current_type, "size": current_size, "is_breaker": previous_p["is_breaker"], "is_heading": previous_p["is_heading"], "page": previous_p["page"], "total_lines": previous_p["total_lines"]})
            start = text
            previous_p = p
            middle = " "
            current_size = size
            hyphen_flag = False
            previous_break = False
            current_type = type
            continue
        if check_captions(text):
            sections.append({"text": start.replace("  ", " "),"type": current_type, "size": current_size, "is_breaker": previous_p["is_breaker"], "is_heading": previous_p["is_heading"], "page": previous_p["page"], "total_lines": previous_p["total_lines"]})
            start = text
            previous_p = p
            middle = " "
            current_size = size
            hyphen_flag = False
            previous_break = False
            current_type = type
            continue
        if current_type != type and not (current_type == type_dict["Nomal"] and type == type_dict["Upper each word"] and start.strip()[-1] not in ["!", "?", ".",":"]):
            sections.append({"text": start.replace("  ", " "),"type": current_type, "size": current_size, "is_breaker": previous_p["is_breaker"], "is_heading": previous_p["is_heading"], "page": previous_p["page"], "total_lines": previous_p["total_lines"]})
            start = text
            previous_p = p
            middle = " "
            current_size = size
            hyphen_flag = False
            previous_break = False
            current_type = type
            continue
        if check_captions(start) and text[0].isalpha() and text[0].isupper():
            sections.append({"text": start.replace("  ", " "),"type": current_type, "size": current_size, "is_breaker": previous_p["is_breaker"], "is_heading": previous_p["is_heading"], "page": previous_p["page"], "total_lines": previous_p["total_lines"]})
            start = text
            previous_p = p
            middle = " "
            current_size = size
            hyphen_flag = False
            previous_break = False
            current_type = type
            continue
            
        start += middle + text
        middle = " "
    return sections


# text = "IP adressess."
# type = get_type(text)
# print()
import unicodedata
def fix_text(data):
    data = unicodedata.normalize("NFKD",data)
    data = data.replace('\xad', '-')
    # data = data.replace('\xa0',' ')
    data = data.replace('\u00ad', '-')
    data = data.replace('\N{SOFT HYPHEN}', '-')
    data = data.replace('\u200b', '')
    return data
unwanted_patterns = r"(objectives|Review Questions|acknowledgement|Title Page|Copyright page|Conference Organization|Key Terms|Copyright and Credits|Table of contents|Index|Appendix|Appendices|Glossary|References|Bibliography|About the author|About the book|About the publisher|Preface|Acknowledgements|Dedication|Introduction|List of Figures|List of Tables|Contents|Contributors|Foreword|Other books you may enjoy|Content|list of contributors|List of abbreviations|List of Illustrations|List of authors)"
def remove_unwated(sections):
    sections_ = []
    current_meta = sections[0]
    current_meta["count"] = 1
    current_text = sections[0]["text"]
    count_heading = 0
    for s in sections:
        if s["is_heading"] != -1:
            count_heading += 1
    if count_heading  == 0:
        for s in sections[1:]:

            if s["type"] == 1 and current_text.strip() not in [""]:
                sections_.append({"text": current_text.strip(), "meta": current_meta})
                current_text = s["text"]
                current_meta = s
                current_meta["name"] = s["text"]
                current_meta["count"] = 1
                continue
            current_text += "\n"+s["text"]
            current_meta["count"] +=1
    else:
        for s in sections[1:]:

            if s["is_heading"] != -1 and current_text.strip() not in [""]:
                sections_.append({"text": current_text.strip(), "meta": current_meta})
                current_text = s["text"]
                current_meta = s
                current_meta["name"] = s["text"]
                current_meta["count"] = 1
                continue
            current_text += "\n"+s["text"]
            current_meta["count"] +=1
    start_index = 0
    sections2 = []
    for s in sections_[start_index:]:
        meta = s["meta"]
        section_name = s["meta"]["text"].strip()
        if re.search(unwanted_patterns, section_name, re.IGNORECASE) or re.search("^[A-Z]$", section_name) :
            continue
        sections2.append(s)
    return sections2
starting_patterns = r"^(Introduction|Chapter|Section|Part)"
starting_pattern2 = r"^(Preface|Foreword|Table of contents|Contents|List of Figures|List of Tables|acknowledgement)"

def find_starting(bookmarks):
    page_ = -1
    current_index = 0
    for b in bookmarks:
        level = b[0]
        page = b[2]
        text = b[1]
        text = remove_chapter_marking(text)
        if re.search(starting_patterns, text , re.IGNORECASE):
            page_ = page
            break
    if page_ != -1:
        return page_
    
    for index in range(len(bookmarks)):
            b = bookmarks[index]
            level = b[0]
            page = b[2]
            text = b[1]
            if re.search(starting_pattern2, remove_chapter_marking(text) , re.IGNORECASE) :
                page_ = page
                current_index = index
                break
    if page_ != -1:
            index = current_index+1
            while index < len(bookmarks):
                b = bookmarks[index]
                level = b[0]
                page = b[2]
                text = b[1]
                if not re.search(unwanted_patterns, text , re.IGNORECASE):
                    page_ = page
                    break
                index += 1
    if page_ != -1:
        return page_
    if page_ == -1:
        for b in bookmarks:
            level = b[0]
            page = b[2]
            text = b[1]
            text = remove_chapter_marking(text)
            if re.search(heading_pattern, text , re.IGNORECASE):
                page_ = page
                break
        if page_ != -1:
            return page_
    if page_ == -1:
        for b in bookmarks:
            level = b[0]
            page = b[2]
            text = b[1]
            if level == 2: #first page with level 2 #level 1 usually include gibberish
                page_ = page
                break
        if page_ != -1:
            return page_
    if page_ == -1:
        return bookmarks[0][2] #first page
    return 0

unwanted_pattern2 = r"^(Chapter\s|Part\s|Section\s)?[0-9IVXLCDM]{1,2}(\.|\:|\,|\-|$)([1-9IVXLCDM]{1,2}(\.|\:|\,|\-)?)*$"
unwanted_pattern3 = r"^(Copyright \u00a9 by|All rights reserved|This book is dedicated to)"
def check_unwanted(text):
    if re.search(unwanted_pattern2, text, re.IGNORECASE):
        return True
    return False
def clean_book(pdf):
    full_bookmarks = get_full_bookmarks(pdf)
    if len(full_bookmarks) == 0:
        print("No bookmarks found")
        return []
    starting_page = find_starting(full_bookmarks)
    level1_bookmarks,current_lvel = get_important_bookmarks(pdf, level = [1]) #this is for section breaking

    texts = get_text(pdf, full_bookmarks, level1_bookmarks, starting_page,current_level= current_lvel)
    new_texts = []
    for index in range(0,len(texts)-1):
        element = texts[index]
        text = element["text"]
        next_element = texts[index+1]
        if re.search(unwanted_pattern2, text, re.IGNORECASE) and element["is_heading"] == -1:
            continue
        new_texts.append(element)
    new_texts.append(texts[-1])
    sections = re_structure(new_texts)
    sections2 = []
    for s in sections:
        if s["text"].strip() in [""]:
            continue
        if check_captions(s["text"].strip()):
            continue

        if re.search(r"^[0-9]{1,2}\s[A-Za-z]{3,4}", s["text"]):
            continue
        if re.search(r"^See\s", s["text"]):
            continue
        if re.search(unwanted_pattern3, s["text"], re.IGNORECASE):
            continue
        sections2.append(s)
    sections3 = re_structure(sections2)
    sections4 = remove_unwated(sections3)
    count_breaker = 0
    for s in sections4:

            if    s["meta"]["is_breaker"]:
                count_breaker += 1
    keyword = "is_breaker"
    if count_breaker < 5:
        keyword = "is_heading"
    sections5 = []
    current_meta = sections4[0]["meta"]
    current_text = sections4[0]["text"]
    for s in sections4[1:]:
        if keyword == "is_heading":
            if s["meta"][keyword]!=-1 and s["meta"][keyword] <= current_lvel+1 and current_text.strip() not in [""]:
                sections5.append({"text": current_text.strip(), "meta": current_meta})
                current_text = s["text"]
                current_meta = s["meta"]
                continue
            current_meta["count"] += s["meta"]["count"]
            current_text += "\n"+ s["text"]
        if keyword == "is_breaker":
            if s["meta"][keyword] and s["meta"][keyword] <=3 and current_text.strip() not in [""]:
            
                sections5.append({"text": current_text.strip(), "meta": current_meta})
                current_text = s["text"]
                current_meta = s["meta"]
                continue
            current_text += "\n" + s["text"] 
            current_meta["count"] += s["meta"]["count"]
    return sections5

def pdf2text(source_file):
    sections = clean_book(source_file)
    if len(sections) == 0:
        print("no section found")
        return
    else:
        return {"sections": sections}