# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import time
import sqlite3


class LottoAnalysis():

    def __init__(self, name):
        self.win_data_set, self.bonus_data_set = self._read_lotto(name)
        self.final_no = len(self.win_data_set)
        self._is_fixed, self._fixed_num = self._fixed_number_()
        self._want_except, self._except_num = self._except_number()
        self.time, self.delete_10 = self._user_input()
        self._number_range_l, self._number_range_u = self._max_min_range_of_lotto_(self.win_data_set)
        self._prob_of_number = self._probability_of_number()
        self._ac_option = self._choice_ac_prob()
        self._same_option = self._same_prev_num_()
        self._even_option = self._even_num_()
        self._low_option = self._low_num_()
        self._prime_option = self._prime_num_()
        self._end_sum_option = self._end_sum_prob()

    def _read_lotto(self, name):
        """
        로또 파일을 읽어들이는 함수
        """
        try:
            conn = sqlite3.connect(name)
            df = pd.read_sql("SELECT * FROM 회차별_당첨정보", conn, index_col='회차')
        except:
            raise ("해당 파일이 존재하는지 확인 해 주세요.\n", sys.exc_info()[0])

        self._all_data = df[df.columns[:8]]
        win_data, bonus_data = df[df.columns[2:8]], df[df.columns[8]]
        win_data, bonus_data = win_data.values, bonus_data.values
        return win_data, bonus_data

    def _user_input(self):
        """
        입력받는 함수
        """
        number_of_time, Non_appear_number_delete = 0, False
        success = False
        while (not success):
            success = True
            try:
                number_of_time = int(input("Q. 뽑을 로또의 개수를 골라주세요. >>"))
                if number_of_time == 0:
                    sys.exit(0)
            except ValueError:
                print("잘못 입력하셨습니다. ", sys.exc_info()[0])
                success = False

        success = False
        while (not success):
            success = True
            Non_appear_number_delete_ask = input("Q. 최근 10주 간 나오지 않은 번호는 제외하시겠습니까? [Y / N] >>")
            if Non_appear_number_delete_ask.lower() == 'y':
                Non_appear_number_delete = True
            elif Non_appear_number_delete_ask.lower() == 'n':
                Non_appear_number_delete = False
            else:
                print("잘못 입력하셨습니다. 다시 입력해 주세요.")
                success = False
        return number_of_time, Non_appear_number_delete

    def _max_min_range_of_lotto_(self, lotto_Data):
        # 로또번호의 합의 범위를 정하는 함수,(fixed)
        sum_number_data = []
        for sum_number in lotto_Data:
            sum_number_data.append(sum(sum_number))
        return [min(sum_number_data), max(sum_number_data)]

    def _probability_of_number(self):
        # 번호의 확률을 구하는 함수. 옵션 5가지를 입력할 것이다.
        def prob_option(opt):
            # option 선택 함수
            number = [0] * 45
            for i in range(self.final_no):
                for j in range(6):
                    number[self.win_data_set[i][j] - 1] += 1
            if opt == 3 or opt == 5:
                for i in range(len(self.bonus_data_set)):
                    number[self.bonus_data_set[i] - 1] += 1
            sum_number = sum(number)
            number_prob = np.array([num / sum_number for num in number])
            # del number,sum_number

            if opt == 4 or opt == 5:
                for i in range(45):
                    temp = (1 / 45) - number_prob[i]
                    number_prob[i] = (1 / 45) + temp
            return number_prob

        name = list(map(str, list(range(1, 46))))
        op1 = [1 / 45] * 45
        op2 = prob_option(2)
        op3 = prob_option(3)
        op4 = prob_option(4)
        op5 = prob_option(5)

        success = False
        num_prob = [0] * 45
        option = 0
        print('Q. 번호마다의 확률을 어떻게 정하시겠습니까?')
        print('\tOption 1: 모든 번호를 균등한 확률로 뽑는다.(ex. 1/45)')
        print('\tOption 2: 로또 데이터의 확률을 따른다. (보너스 제외)')
        print('\tOption 3: 로또 데이터의 확률을 따른다. (보너스 포함)')
        print(
            '\tOption 4: 많이 나온 숫자에는 가중치를 작게 적게 나온 숫자에는 가중치를 크게 준다.(보너스 제외)\n\t\t(ex. 1이 만약 현재까지 적게 나왔으면 1이 뽑힐 확률을 높인다.)')
        print(
            '\tOption 5: 많이 나온 숫자에는 가중치를 작게 적게 나온 숫자에는 가중치를 크게 준다.(보너스 포함)\n\t\t(ex. 1이 만약 현재까지 적게 나왔으면 1이 뽑힐 확률을 높인다.)')
        # 크기를 변경하고 싶으면 fiqur.figsize 사용
        plt.rcParams["figure.figsize"] = (18, 16)
        plt.title("Number probability")
        plt.xlabel("Number")
        plt.ylabel("Probability")
        plt.grid(True)
        plt.plot(name, op1, 'b*--', label="Option1")
        plt.plot(name, op2, 'r.--', label="Option2")
        plt.plot(name, op3, 'g^--', label="Option3")
        plt.plot(name, op4, 'yv--', label="Option4")
        plt.plot(name, op5, 'k+--', label="Option5")
        plt.legend(loc="best")
        plt.show()
        while (not success):
            success = True
            try:
                option = int(input('옵션을 선택해 주세요. (1, 2, 3, 4, 5) >>'))
                if option > 5 or option < 1:
                    print('번호를 잘못 입력하셨습니다.')
                    success = False
            except ValueError:
                print('잘못 입력하셨습니다.')
                success = False

        self._option_num_select = option

        if option == 1:
            num_prob = np.array([1 / 45] * 45)
        else:
            num_prob = prob_option(option)

        return num_prob

    def _choose_number(self):
        # 번호 선택 함수
        return np.random.choice(np.arange(1, 46), p=self._prob_of_number)

    def _is_same_prev(self, pred_lotto_list):
        # 예측값이 이전에 있었는지를 물어보는 함수.
        for prev_list in self._all_data:
            if len(np.setdiff1d(prev_list, pred_lotto_list)) == 0 or len(np.setdiff1d(prev_list, pred_lotto_list)) == 1:
                return True
        return False

    def _get_ac_of_list(self, lotto_list):
        # 6자리 리스트의 ac값을 반환하는 함수
        dist = []
        for i in range(len(lotto_list)):
            for j in range(0, i):
                dist.append(abs(lotto_list[i] - lotto_list[j]))
        dist = list(set(dist))
        return len(dist) - 5

    def _choice_ac_prob(self):
        # ac의 옵션을 설정하는 함수
        name_x = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        origin_ac = [180, 858, 7873, 26292, 103461, 239998, 823318, 1148170, 2675680, 1609174, 1510056]
        origin_ac_prob = np.array([num / sum(origin_ac) for num in origin_ac])
        lotto_ac_data = [0] * 11
        weights_of_prob_ac_data = [0] * 11
        for lotto_data in self.win_data_set:
            lotto_ac_data[self._get_ac_of_list(lotto_data)] += 1

        weights_of_num_ac_data = [origin_ac[i] - lotto_ac_data[i] for i in range(len(origin_ac))]
        sum_weight_num = sum(weights_of_num_ac_data)
        sum_lotto_ac = sum(lotto_ac_data)
        for i in range(11):
            lotto_ac_data[i] /= sum_lotto_ac
            weights_of_num_ac_data[i] /= sum_weight_num
            temp = origin_ac_prob[i] - lotto_ac_data[i]
            if (origin_ac_prob[i] + temp) >= 0:
                weights_of_prob_ac_data[i] = origin_ac_prob[i] + temp

        plt.rcParams["figure.figsize"] = (10, 4)
        plt.title("Probability compare")
        plt.bar(name_x, origin_ac_prob, label='Probability AC Origin')
        plt.plot(name_x, lotto_ac_data, 'r.--', label='Probability AC Lotto')
        plt.plot(name_x, weights_of_num_ac_data, 'k.--', label='Probability AC num')
        plt.plot(name_x, weights_of_prob_ac_data, 'y.--', label='Probability AC prob')
        plt.xlabel('AC value')
        plt.ylabel('Probability')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()
        option = 0
        success = False
        print("Q. AC값을 어떻게 정하시겠습니까?")
        print("\tOption 1: 전체조합의 AC확률을 따르겠다.(파란색 막대그래프)")
        print("\tOption 2: 로또조합의 AC확률을 따르겠다.(빨간색 점선)")
        print("\tOption 3: 가중치를 숫자에 주겠다. (검은색 점선)")
        print("\tOption 4: 가중치를 확률에 주겠다. (노란색 점선)")
        print("\tOption 5: AC값을 고정하겠다.")
        print("\tOption 6: AC값의 범위를 주어주겠다.")
        while (not success):
            success = True
            try:
                option = int(input('옵션을 선택해 주세요. (1, 2, 3, 4, 5, 6) >>'))
                if option > 6 or option < 1:
                    print('번호를 잘못 입력하셨습니다.')
                    success = False
            except ValueError:
                print('잘못 입력하셨습니다.')
                success = False
        self._option_ac_select = option
        # option 방법 중고정은 0 범위는 1 확률은 2
        if option == 5:
            success = False
            while (not success):
                success = True
                try:
                    input_ac = int(input('고정할 AC값을 입력해주세요. (0 ~ 10) >>'))
                    if input_ac > 10 or input_ac < 0:
                        print('0 ~ 10 숫자를 입력해주세요.')
                        success = False
                except ValueError:
                    print('잘못 입력하셨습니다.')
                    success = False
            self._ac = input_ac
            return 0
        elif option == 6:
            upper_ac, lower_ac = 0, 0
            success = False
            while (not success):
                success = True
                try:
                    lower_ac = int(input('하한 AC값을 입력해주세요. (0 ~ 10) >>'))
                    if lower_ac > 10 or lower_ac < 0:
                        print('0 ~ 10 숫자를 입력해주세요.')
                        success = False
                except ValueError:
                    print('잘못 입력하셨습니다.')
                    success = False

                try:
                    upper_ac = int(input('상한 AC값을 입력해주세요. (0 ~ 10) >>'))
                    if upper_ac > 10 or upper_ac < 0:
                        print('0 ~ 10 숫자를 입력해주세요.')
                        success = False
                except ValueError:
                    print('잘못 입력하셨습니다.')
                    success = False

                if lower_ac > upper_ac:
                    print("상한 AC: " + str(upper_ac) + " 보다 하한 AC: " + str(lower_ac) + "가 더 크므로 자동으로 변환하겠습니다.")
                    lower_ac, upper_ac = upper_ac, lower_ac

                elif lower_ac == upper_ac:
                    print("상한 AC와 하한 AC가 동일하므로 AC = " + str(lower_ac) + "로 고정하겠습니다.")
                    self._option_ac_select = 5
                    self._ac = lower_ac

            self._ac = lower_ac, upper_ac
            return 1

        else:
            if option == 1:
                self._ac_prob = origin_ac_prob
            elif option == 2:
                self._ac_prob = lotto_ac_data
            elif option == 3:
                self._ac_prob = weights_of_num_ac_data
            else:
                self._ac_prob = weights_of_prob_ac_data
            return 2

    def _10_none_appear_num(self):
        num = [0] * 45
        for i in range(self.final_no - 10, self.final_no):
            for j in range(6):
                num[self.win_data_set[i][j] - 1] += 1
            num[self.bonus_data_set[i] - 1] += 1
        zero_num = []
        for i in range(num):
            if num[i] == 0:
                zero_num.append(i + 1)
        return zero_num

    def _fixed_number_(self):
        # 고정 시킬 번호 선택
        success = False
        fix_nums = []
        while (not success):
            success = True
            want_fix = input("Q. 반드시 고정시킬 숫자가 있습니까? [Y / N] >>")
            if want_fix.lower() == 'y':
                success2 = False
                while (not success2):
                    success2 = True
                    try:
                        fix_nums = input("공백으로 구분하여 고정시킬 숫자를 입력해주세요. (ex. 1 2 3)").split()
                        fix_nums = list(map(int, fix_nums))
                    except ValueError:
                        print('잘못 입력하셨습니다.')
                        success2 = False
                    fix_nums.sort()
                    if fix_nums[0] < 1 or fix_nums[-1] > 45 or len(fix_nums) > 6:
                        print("1 ~ 45까지의 수를 6개 이하로 입력해주세요.")
                        success2 = False
                if len(fix_nums) == 6:
                    print(fix_nums)
                    sys.exit(1)
                return True, fix_nums
            elif want_fix.lower() == 'n':
                return False, fix_nums

            else:
                print("잘못 입력하셨습니다. 다시 입력해 주세요.")
                success = False

    def _except_number(self):
        # 제외 시킬 번호 선택
        success = False
        except_nums = None
        while (not success):
            success = True
            want_except = input("Q. 반드시 제외시킬 숫자가 있습니까? [Y / N] >>")
            if want_except.lower() == 'y':
                success2 = False
                while (not success2):
                    success2 = True
                    try:
                        except_nums = input("공백으로 구분하여 제외시킬 숫자를 입력해주세요. (ex. 1 2 3)").split()
                        except_nums = list(map(int, except_nums))
                    except ValueError:
                        print('잘못 입력하셨습니다.')
                        success2 = False
                    except_nums.sort()
                    if except_nums[0] < 1 or except_nums[-1] > 45:
                        print("1 ~ 45까지의 수를 입력해주세요.")
                        success2 = False
                if len(except_nums) + 6 > 45:
                    print("만족하는 번호가 존재하지 않습니다.")
                    sys.exit(1)
                elif len(except_nums) + 6 == 45:
                    print(np.setdiff1d(list(range(1, 46)), except_nums))
                    sys.exit(1)
                return True, except_nums
            elif want_except.lower() == 'n':
                return False, None

            else:
                print("잘못 입력하셨습니다. 다시 입력해 주세요.")
                success = False

    def _same_prev_num_(self):
        # 전 회차와 똑같은 번호의 개수 option 2개
        same_prev = [0] * 7
        name = ['0', '1', '2', '3', '4', '5', '6']
        for i in range(1, self.final_no):
            same_prev[len(np.intersect1d(self.win_data_set[i - 1], self.win_data_set[i]))] += 1
        # print(same_prev)
        sum_same = sum(same_prev)
        same_prev = [num / sum_same for num in same_prev]
        # print(same_prev)

        plt.title("Same prev probability")
        plt.grid(True)
        plt.bar(name, same_prev)
        plt.xlabel('Number')
        plt.ylabel('Probability')
        plt.show()
        option = 0
        print("Q. 전 회차와 똑같은 번호의 개수를 어떻게 정하시겠습니까?")
        print("\tOption 1: 직접 입력을 한다.")
        print("\tOption 2: 로또조합의 확률을 따르겠다.(파란색 그래프)")
        success = False
        num_same = 0
        while (not success):
            success = True
            try:
                option = int(input("옵션을 선택해주세요. (1 ~ 2) >>"))
                if option < 1 or option > 2:
                    print("1번과 2번 중 선택해주세요.")
                    success = False
            except:
                print("잘못 입력하셨습니다. 다시 입력해주세요.")
                success = False
            if option == 1:
                success2 = False
                while (not success2):
                    success2 = True
                    try:
                        num_same = int(input("개수를 입력해 주세요. (0 ~ 6) >>"))
                        if num_same > 6 or num_same < 0:
                            print("0 ~ 6의 자연수를 입력해주세요.")
                            success2 = False
                    except:
                        print("잘못 입력하셨습니다. 다시 입력해주세요.")
                        success2 = False
                if num_same == 6:
                    print(self.win_data_set[self.final_no - 1])
                    sys.exit(1)
                self._same_prev_num = num_same
                return 1
            elif option == 2:
                self._same_prev_num_prob = same_prev
                return 2

    def _even_num_(self):
        # 짝수의 개수를 정하는 함수
        def isEven(num):
            if num % 2 == 0:
                return True
            return False

        num_even = [0] * 7
        name = ['0', '1', '2', '3', '4', '5', '6']
        for lotto_list in self.win_data_set:
            count = 0
            for lotto_num in lotto_list:
                if isEven(lotto_num):
                    count += 1
                num_even[count] += 1
        sum_even = sum(num_even)
        num_even = [num / sum_even for num in num_even]
        plt.title("Even number probability")
        plt.grid(True)
        plt.bar(name, num_even)
        plt.xlabel("Number")
        plt.ylabel("Probability")
        plt.show()
        option = 0
        print("Q. 짝수의 개수를 어떻게 정하시겠습니까?")
        print("\tOption 1: 직접 입력을 한다.")
        print("\tOption 2: 로또조합의 확률을 따르겠다.(파란색 그래프)")
        success = False
        num_even_sel = 0
        while (not success):
            success = True
            try:
                option = int(input("옵션을 선택해주세요. (1 ~ 2) >>"))
                if option < 1 or option > 2:
                    print("1번과 2번 중 선택해주세요.")
                    success = False
            except:
                print("잘못 입력하셨습니다. 다시 입력해주세요.")
                success = False
            if option == 1:
                success2 = False
                while (not success2):
                    success2 = True
                    try:
                        num_even_sel = int(input("개수를 입력해 주세요. (0 ~ 6) >>"))
                        if num_even_sel > 6 or num_even_sel < 0:
                            print("0 ~ 6의 자연수를 입력해주세요.")
                            success2 = False
                    except:
                        print("잘못 입력하셨습니다. 다시 입력해주세요.")
                        success2 = False

                self._even_num = num_even_sel
                return 1
            elif option == 2:
                self._even_num_prob = num_even
                return 2

    def _low_num_(self):
        # 23보다 작은 정수의 개수를 정하는 함수
        def isLow(num):
            if num < 23:
                return True
            return False

        num_low = [0] * 7
        name = ['0', '1', '2', '3', '4', '5', '6']
        for lotto_list in self.win_data_set:
            count = 0
            for lotto_num in lotto_list:
                if isLow(lotto_num):
                    count += 1
                num_low[count] += 1
        sum_low = sum(num_low)
        num_low = [num / sum_low for num in num_low]
        plt.title("Low(1 ~ 22) number probability")
        plt.grid(True)
        plt.bar(name, num_low)
        plt.xlabel("Number")
        plt.ylabel("Probability")
        plt.show()
        option = 0
        print("Q. 23보다 작은 번호의 개수를 어떻게 정하시겠습니까?")
        print("\tOption 1: 직접 입력을 한다.")
        print("\tOption 2: 로또조합의 확률을 따르겠다.(파란색 그래프)")
        success = False
        num_low_sel = 0
        while (not success):
            success = True
            try:
                option = int(input("옵션을 선택해주세요. (1 ~ 2) >>"))
                if option < 1 or option > 2:
                    print("1번과 2번 중 선택해주세요.")
                    success = False
            except:
                print("잘못 입력하셨습니다. 다시 입력해주세요.")
                success = False
            if option == 1:
                success2 = False
                while (not success2):
                    success2 = True
                    try:
                        num_low_sel = int(input("개수를 입력해 주세요. (0 ~ 6) >>"))
                        if num_low_sel > 6 or num_low_sel < 0:
                            print("0 ~ 6의 자연수를 입력해주세요.")
                            success2 = False
                    except:
                        print("잘못 입력하셨습니다. 다시 입력해주세요.")
                        success2 = False

                self._low_num = num_low_sel
                return 1
            elif option == 2:
                self._low_num_prob = num_low
                return 2

    def _prime_num_(self):
        # 소수의 개수를 정하는 함수
        def isPrime(num):
            for i in range(num):
                if num == 1:
                    return False
                else:
                    for i in range(2, num):
                        if num % i == 0:
                            return False
            return True

        num_prime = [0] * 7
        name = ['0', '1', '2', '3', '4', '5', '6']
        for lotto_list in self.win_data_set:
            count = 0
            for lotto_num in lotto_list:
                if isPrime(lotto_num):
                    count += 1
                num_prime[count] += 1
        sum_prime = sum(num_prime)
        num_prime = [num / sum_prime for num in num_prime]
        plt.title("Prime number probability")
        plt.grid(True)
        plt.bar(name, num_prime)
        plt.xlabel("Number")
        plt.ylabel("Probability")
        plt.show()
        option = 0
        print("Q. 소수의 개수를 어떻게 정하시겠습니까?")
        print("\tOption 1: 직접 입력을 한다.")
        print("\tOption 2: 로또조합의 확률을 따르겠다.(파란색 그래프)")
        success = False
        num_prime_sel = 0
        while (not success):
            success = True
            try:
                option = int(input("옵션을 선택해주세요. (1 ~ 2) >>"))
                if option < 1 or option > 2:
                    print("1번과 2번 중 선택해주세요.")
                    success = False
            except:
                print("잘못 입력하셨습니다. 다시 입력해주세요.")
                success = False
            if option == 1:
                success2 = False
                while (not success2):
                    success2 = True
                    try:
                        num_prime_sel = int(input("개수를 입력해 주세요. (0 ~ 6) >>"))
                        if num_prime_sel > 6 or num_prime_sel < 0:
                            print("0 ~ 6의 자연수를 입력해주세요.")
                            success2 = False
                    except:
                        print("잘못 입력하셨습니다. 다시 입력해주세요.")
                        success2 = False

                self._prime_num = num_prime_sel
                return 1
            elif option == 2:
                self._prime_num_prob = num_prime
                return 2

    def _end_sum(self, lotto_list):
        sum_End = 0
        for num_lotto in lotto_list:
            sum_End += (num_lotto % 10)
        return sum_End

    def _end_num_sum_range(self, end_sum):
        if end_sum in range(5, 11):
            return 0
        elif end_sum in range(11, 16):
            return 1
        elif end_sum in range(16, 21):
            return 2
        elif end_sum in range(21, 26):
            return 3
        elif end_sum in range(26, 31):
            return 4
        elif end_sum in range(31, 36):
            return 5
        elif end_sum in range(36, 41):
            return 6
        elif end_sum in range(41, 46):
            return 7
        elif end_sum in range(46, 51):
            return 8
        elif end_sum in range(51, 56):
            return 9
        else:
            raise ("cannot find end_sum_range")

    def _end_sum_prob(self):
        name = np.array(['5~10', '11~15', '16~20', '21~25', '26~30', '31~35', '36~40', '41~45', '46~50', '51~55'])
        end_sum_range = [0] * 10
        for lotto_list in self.win_data_set:
            end_sum_range[self._end_num_sum_range(self._end_sum(lotto_list))] += 1
        sum_end_sum_range = sum(end_sum_range)
        # print(end_sum_range)
        end_sum_range = [num_lotto / sum_end_sum_range for num_lotto in end_sum_range]
        plt.title("End sum range probability")
        plt.grid(True)
        plt.bar(name, end_sum_range)
        plt.xlabel("Range")
        plt.ylabel("Probability")
        plt.show()
        option = 0
        print("Q. 끝수합의 범위를 어떻게 정하시겠습니까?")
        print("\tOption 1: 범위를 직접 입력을 한다.")
        print("\tOption 2: 로또조합의 확률을 따르겠다.(파란색 그래프)")
        success = False
        num_end_sum_sel = 0
        while (not success):
            success = True
            try:
                option = int(input("옵션을 선택해주세요. (1 ~ 2) >>"))
                if option < 1 or option > 2:
                    print("1번과 2번 중 선택해주세요.")
                    success = False
            except:
                print("잘못 입력하셨습니다. 다시 입력해주세요.")
                success = False
            if option == 1:
                success2 = False
                while (not success2):
                    success2 = True
                    try:
                        num_end_sum_sel = input("범위를 공백으로 구분하여 입력해 주세요. (ex. 5 10)\n(자동으로 순서가 정렬됩니다.) >>").split()
                        if len(num_end_sum_sel) != 2:
                            print('두개의 수를 입력해 주세요.')
                            success2 = False
                        else:
                            num_end_sum_sel = list(map(int, num_end_sum_sel))
                            num_end_sum_sel.sort()
                            if num_end_sum_sel[0] < 5 or num_end_sum_sel[-1] > 55:
                                print("5 ~ 55의 자연수를 입력해주세요.")
                                success2 = False
                    except:
                        print("잘못 입력하셨습니다. 다시 입력해주세요.")
                        success2 = False

                self._end_sum_range = num_end_sum_sel
                return 1
            elif option == 2:
                self._end_sum_range_prob = end_sum_range
                return 2

    def _change_value_(self):
        # print("chainging Value")
        if self._ac_option == 2:
            # print("chainging Value ac")
            if sum(self._ac_prob) > 1:
                self._ac_prob = self._ac_prob / sum(self._ac_prob)
            self._ac = np.random.choice(np.arange(0, 11), p=self._ac_prob)
        if self._even_option == 2:
            # print("chainging Value even")
            self._even_num = np.random.choice(np.arange(0, 7), p=self._even_num_prob)
        if self._low_option == 2:
            # print("chainging Value low")
            self._low_num = np.random.choice(np.arange(0, 7), p=self._low_num_prob)
        if self._prime_option == 2:
            # print("chainging Value prime")
            self._prime_num = np.random.choice(np.arange(0, 7), p=self._prime_num_prob)
        if self._end_sum_option == 2:
            # print("chainging Value end")
            self._end_sum_range = np.random.choice(np.arange(0, 10), p=self._end_sum_range_prob)
        if self._same_option == 2:
            # print("chainging Value smae")
            self._same_prev_num = np.random.choice(np.arange(0, 7), p=self._same_prev_num_prob)
        return None

    def _find_the_condition(self, lotto_list):
        # 조건 확인
        def isPrime(num):
            for i in range(num):
                if num == 1:
                    return False
                else:
                    for i in range(2, num):
                        if num % i == 0:
                            return False
            return True

        # 소수 검색
        def isLow(num):
            if num < 23:
                return True
            return False

        # 23보다 작은 수 검색
        def isEven(num):
            if num % 2 == 0:
                return True
            return False

        # 짝수 검색
        prime, even, low, ac, end_sum, same_prev = 0, 0, 0, 0, 0, 0
        for num in lotto_list:
            if isPrime(num):
                prime += 1
            if isEven(num):
                even += 1
            if isLow(num):
                low += 1
        ac = self._get_ac_of_list(lotto_list)
        end_sum = self._end_sum(lotto_list)
        same_prev = len(np.intersect1d(self.win_data_set[self.final_no - 1], lotto_list))
        if sum(lotto_list) > self._number_range_u or sum(lotto_list) < self._number_range_l:
            # print("sum Error")
            return False
        if self._ac_option == 1:
            if ac < min(self._ac) or ac > max(self._ac):
                # print("AC Option 2 Error")
                return False
        else:
            if ac != self._ac:
                # print("AC option 1 | 3 Error")
                return False
        if even != self._even_num or low != self._low_num or prime != self._prime_num or same_prev != self._same_prev_num:
            # print("Even, Low, Prime Error")
            return False
        if self._end_sum_option == 1:
            if end_sum < min(self._end_sum_range) or end_sum > max(self._end_sum_range):
                # print("End Sum Option1 Error")
                return False
        else:
            if self._end_num_sum_range(end_sum) != self._end_sum_range:
                # print("End sum option2 Error")
                return False
        # print("I find")
        return True

    def run_analysis(self):
        lotto = []
        if self._is_fixed:
            lotto = self._fixed_num
        delete_num = []
        if self._want_except:
            delete_num = self._except_num

        def get_list(lotto_list):
            # print("run_get_list_676")
            ac = self._get_ac_of_list(lotto_list)
            same_prev = len(np.intersect1d(self.win_data_set[self.final_no - 1], lotto_list))
            end_sum = self._end_sum(lotto_list)
            return ac, same_prev, end_sum

        for i in range(self.time):
            start = time.time()
            print(">> " + str(i + 1) + " 번째 번호 추출 중...")
            # self._change_value_()
            find = False
            iteration = 0
            while (not find):
                find = True
                iteration += 1
                get_lotto = lotto.copy()
                self._change_value_()
                while len(get_lotto) < 6:
                    number = self._choose_number()
                    if number in get_lotto or number in delete_num:
                        continue
                    get_lotto.append(number)
                get_lotto.sort()
                if len(get_lotto) != 6:
                    raise ("Wrong here")
                find = self._find_the_condition(get_lotto)
                if find:
                    print(">> " + str(i + 1) + " 번째 번호 추출 완료!")
                    ac_value, sampe_prev, end_sum_value = get_list(get_lotto)
                    print(">> Lotto: {}".format(get_lotto))
                    print(">> iteration: {}, AC = {}, same Prev = {}, end sum = {}".format(iteration, ac_value,
                                                                                           sampe_prev, end_sum_value))
                    print("---%s seconds---" % (time.time() - start))

