class predictor:
    def init(self, model_path, meta_path):


        folder_path = r'..\data\raw'
        file_type = '/*csv'
        files = glob.glob(folder_path + file_type)
        latest_file = max(files, key=os.path.getctime)

        time = latest_file[12:].split(sep="_")

        self.access_time = pd.Timestamp(
            year=int(time[0][0:4]),
            month=int(time[0][4:6]),
            day=int(time[0][6:8]),
            hour=int(time[1][0:2]),
            minute=int(time[1][2:4]),
            tz='Europe/Moscow'
        )

    def predict(self, ad_dict) -> dict:
        pass

    def get_meta(self) -> dict:
        pass