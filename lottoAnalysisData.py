# -*- coding: utf-8 -*-
import pandas as pd
import requests
import sqlite3
from bs4 import BeautifulSoup
import sys


class UpdateDB():

    def __init__(self, db_file):
        self.main_url = "https://dhlottery.co.kr/gameResult.do?method=byWin"
        self.basic_url = "https://dhlottery.co.kr/gameResult.do?method=byWin&drwNo="
        self._db = db_file
        self.start_no = 0
        self.lotto_list = []
        self.last = self._GetLast()
        self._need_update_data(db_file)

    def _GetLast(self):
        resp = requests.get(self.main_url)
        soup = BeautifulSoup(resp.text, "html.parser")
        result = str(soup.find("meta", {"id": "desc", "name": "description"})['content'])

        s_idx = result.find(" ")
        e_idx = result.find("회")
        return int(result[s_idx + 1: e_idx])

    def _need_update_data(self, name):
        try:
            conn = sqlite3.connect(name)
            df = pd.read_sql("SELECT * FROM 회차별_당첨정보", conn, index_col='회차')
        except:
            raise ("해당 파일이 존재하는지 확인 해 주세요.\n", sys.exc_info()[0])
        now_Ver = len(df)
        isOk = False
        if now_Ver < self.last:
            while not isOk:
                isOk = True
                select = input("현재 데이터({}회)가 최신({}회)이 아닙니다. 업데이트 하시겠습니까?[Y / N] >>>".format(now_Ver, self.last))
                if select.lower() == 'y':
                    print("Update start")
                    self._Crawler(now_Ver, self.last)
                    self.Data_to_DB()
                    print("Update Complete")
                elif select.lower() == 'n':
                    continue
                else:
                    print("잘못입력하셨습니다. 다시 입력해주세요.")
                    isOk = False
        else:
            print("최신버전을 사용하고 있습니다.")

    def _Crawler(self, start, end):
        for i in range(start, end + 1):
            print("Data {}th extraction...".format(i))
            crawler_url = self.basic_url + str(i)

            resp = requests.get(crawler_url)
            soup = BeautifulSoup(resp.text, "html.parser")
            result = str(soup.find("meta", {"id": "desc", "name": "description"})['content'])

            s_idx = result.find("당첨번호")
            s_idx = result.find(" ", s_idx) + 1
            e_idx = result.find(".", s_idx)
            numbers = result[s_idx:e_idx]

            s_idx = result.find("총")
            s_idx = result.find(" ", s_idx) + 1
            e_idx = result.find("명", s_idx)
            persons = result[s_idx:e_idx]

            s_idx = result.find("당첨금액")
            s_idx = result.find(" ", s_idx) + 1
            e_idx = result.find("원", s_idx)
            price = result[s_idx:e_idx]

            info = {}
            info["회차"] = i
            info["당첨번호"] = numbers
            info["당첨인원"] = persons
            info["당첨금액"] = price
            # print(info)
            self.lotto_list.append(info)

    def Data_to_DB(self):
        print("Data saving...")
        persons = []
        price = []
        con = sqlite3.connect(self._db)

        for data_i in self.lotto_list:
            count = data_i["회차"]
            numbers = data_i["당첨번호"]
            persons = data_i["당첨인원"]
            price = data_i["당첨금액"]

            numberlist = str(numbers).split(",")

            sql = "INSERT INTO '회차별_당첨정보'('회차','1','2','3','4','5','6','bonus','당첨자수','당첨금액')" \
                  "VALUES(?,?,?,?,?,?,?,?,?,?)"
            data = (count,
                    int(numberlist[0]),
                    int(numberlist[1]),
                    int(numberlist[2]),
                    int(numberlist[3]),
                    int(numberlist[4]),
                    int(numberlist[5].split("+")[0]),
                    int(numberlist[5].split("+")[1]),
                    int(persons),
                    str(price))
            cur = con.cursor()
            cur.execute(sql, data)
            con.commit()

            sql = "DELETE FROM '회차별_당첨정보' WHERE rowid NOT IN (SELECT Max(rowid) FROM '회차별_당첨정보' GROUP BY 회차 order by 회차)"
            cur.execute(sql)
            con.commit()
        con.close()
        print("Finish saving")
