# -*- coding: utf-8 -*-

from lottoAnalysis import LottoAnalysis as LA
from lottoAnalysisData import UpdateDB as UBD
if __name__ == "__main__":
    db_file = "lotto_data.db"
    database = UBD(db_file)
    Lotto = LA(db_file)
    Lotto.run_analysis()