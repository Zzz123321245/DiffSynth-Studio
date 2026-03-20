from .operators import *
import os, torch, json, pandas, random, time


class UnifiedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        repeat=1,
        data_file_keys=tuple(),
        main_data_operator=lambda x: x,
        special_operator_map=None,
        max_data_items=None,
        dataset_min_num_frames=0,
        dataset_num_frames_key="num_frames",
        dataset_drop_missing_num_frames=False,
        dataset_read_retry_count=0,
        dataset_read_retry_sleep_seconds=0.0,
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.data_file_keys = data_file_keys
        self.main_data_operator = main_data_operator
        self.cached_data_operator = LoadTorchPickle()
        self.special_operator_map = {} if special_operator_map is None else special_operator_map
        self.max_data_items = max_data_items
        self.dataset_min_num_frames = dataset_min_num_frames
        self.dataset_num_frames_key = dataset_num_frames_key
        self.dataset_drop_missing_num_frames = dataset_drop_missing_num_frames
        self.dataset_read_retry_count = max(int(dataset_read_retry_count), 0)
        self.dataset_read_retry_sleep_seconds = max(float(dataset_read_retry_sleep_seconds), 0.0)
        self.data = []
        self.cached_data = []
        self.load_from_cache = metadata_path is None
        self.load_metadata(metadata_path)

    @staticmethod
    def default_image_operator(
        base_path="",
        max_pixels=1920 * 1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor)),
            (list, SequencialProcess(ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor))),
        ])

    @staticmethod
    def default_video_operator(
        base_path="",
        max_pixels=1920 * 1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, time_division_factor=4, time_division_remainder=1,
        frame_rate=24, fix_frame_rate=False,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
                (("jpg", "jpeg", "png", "webp"), LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor) >> ToList()),
                (("gif",), LoadGIF(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
                (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideo(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                    frame_rate=frame_rate, fix_frame_rate=fix_frame_rate,
                )),
            ])),
        ])

    def search_for_cached_data_files(self, path):
        for file_name in os.listdir(path):
            subpath = os.path.join(path, file_name)
            if os.path.isdir(subpath):
                self.search_for_cached_data_files(subpath)
            elif subpath.endswith(".pth"):
                self.cached_data.append(subpath)

    @staticmethod
    def parse_num_frames_value(value):
        if value is None:
            return None
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    def filter_metadata_by_num_frames(self):
        if self.load_from_cache or self.dataset_min_num_frames <= 0:
            return

        total_before = len(self.data)
        if total_before == 0:
            return

        kept_data = []
        dropped_short = 0
        dropped_missing_or_invalid = 0

        for row in self.data:
            num_frames_value = self.parse_num_frames_value(row.get(self.dataset_num_frames_key))

            if num_frames_value is None:
                if self.dataset_drop_missing_num_frames:
                    dropped_missing_or_invalid += 1
                    continue
                kept_data.append(row)
                continue

            if num_frames_value < self.dataset_min_num_frames:
                dropped_short += 1
                continue

            kept_data.append(row)

        self.data = kept_data
        total_after = len(self.data)

        print(
            f"Metadata frame filter enabled: {self.dataset_num_frames_key}>={self.dataset_min_num_frames}. "
            f"before={total_before}, kept={total_after}, "
            f"dropped_short={dropped_short}, dropped_missing_or_invalid={dropped_missing_or_invalid}."
        )

        if total_after == 0:
            raise ValueError(
                "No data left after frame-count filtering. "
                "Please lower --dataset_min_num_frames or disable filtering."
            )

    def load_metadata(self, metadata_path):
        if metadata_path is None:
            print("No metadata_path. Searching for cached data files.")
            self.search_for_cached_data_files(self.base_path)
            print(f"{len(self.cached_data)} cached data files found.")
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in f:
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            metadata = pandas.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

        self.filter_metadata_by_num_frames()

    def _get_data_source_size(self):
        return len(self.cached_data) if self.load_from_cache else len(self.data)

    def _get_sample_summary(self, data_id):
        if self.load_from_cache:
            return self.cached_data[data_id % len(self.cached_data)]

        row = self.data[data_id % len(self.data)]
        if isinstance(row, dict):
            for key in self.data_file_keys:
                if key in row:
                    return f"{key}={row[key]}"
        return str(row)

    def _load_item_once(self, data_id):
        if self.load_from_cache:
            data = self.cached_data[data_id % len(self.cached_data)]
            data = self.cached_data_operator(data)
        else:
            data = self.data[data_id % len(self.data)].copy()
            for key in self.data_file_keys:
                if key in data:
                    if key in self.special_operator_map:
                        data[key] = self.special_operator_map[key](data[key])
                    elif key in self.data_file_keys:
                        data[key] = self.main_data_operator(data[key])
        return data

    def __getitem__(self, data_id):
        retry_count = self.dataset_read_retry_count if not self.load_from_cache else 0
        num_attempts = retry_count + 1
        last_exception = None
        attempted_indices = []
        data_source_size = self._get_data_source_size()

        for attempt_id in range(num_attempts):
            if attempt_id == 0 or data_source_size <= 1:
                candidate_data_id = data_id
            else:
                candidate_data_id = random.randrange(data_source_size)
            candidate_index = candidate_data_id % data_source_size

            try:
                return self._load_item_once(candidate_data_id)
            except Exception as exc:
                last_exception = exc
                attempted_indices.append(candidate_index)
                sample_summary = self._get_sample_summary(candidate_index)
                print(
                    f"[WARN] Failed to load dataset sample idx={candidate_index} "
                    f"({sample_summary}) attempt={attempt_id + 1}/{num_attempts}: "
                    f"{type(exc).__name__}: {exc}"
                )
                if (
                    attempt_id + 1 < num_attempts
                    and self.dataset_read_retry_sleep_seconds > 0
                ):
                    time.sleep(self.dataset_read_retry_sleep_seconds)

        raise RuntimeError(
            "Failed to load dataset sample after retries. "
            f"attempted_indices={attempted_indices}"
        ) from last_exception

    def __len__(self):
        if self.max_data_items is not None:
            return self.max_data_items
        elif self.load_from_cache:
            return len(self.cached_data) * self.repeat
        else:
            return len(self.data) * self.repeat

    def check_data_equal(self, data1, data2):
        # Debug only
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True
