from flask import Flask, abort
from flask_restful import Resource, reqparse, Api

from tensorbay.dataset import Dataset, FusionDataset, Data

app = Flask(__name__)
api = Api(app)


def vision(local_dataset):
    global dataset
    global dataset_type
    dataset = local_dataset
    dataset_type = 0 if isinstance(dataset, Dataset) else 1

    if not isinstance(dataset, (Dataset, FusionDataset)):
        raise TypeError("It is not a Dataset.")
    app.run()


class SegmentList(Resource):
    def __init__(self):
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument('limit', type=int, required=False, default=128,
                                   help='Limit should be int.', location='args')
        self.reqparse.add_argument('offset', type=int, required=False, default=0,
                                   help='Offset should be int.', location='args')
        self.reqparse.add_argument('sortBy', type=str, required=False, default="asc",
                                   help='SortBy should be str.', location='args')

    def get(self):
        args = self.reqparse.parse_args()
        limit = args["limit"]
        offset = args["offset"]
        sort_by = args["sortBy"]

        response = {"segments": []}
        for segment in dataset[offset: offset + limit]:
            response_segment = {
                "name": segment.name,
                "description": segment.description
            }
            response["segments"].append(response_segment)

        response["offset"] = offset
        response["recordSize"] = len(response["segments"])
        response["totalCount"] = len(dataset)

        if sort_by == "desc":
            response["segments"].reverse()
        return response


class Catalog(Resource):
    def get(self):
        return {"catalogs": dataset.catalog.dumps()}


class DataUriList(Resource):
    def __init__(self):
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument('segmentName', type=str, required=False, default="",
                                   help='SegmentName should be str.', location='args')
        self.reqparse.add_argument('remotePath	', type=str, required=False,
                                   help='RemotePath should be str.', location='args')
        self.reqparse.add_argument('limit', type=int, required=False, default=128,
                                   help='Limit should be int.', location='args')
        self.reqparse.add_argument('offset', type=int, required=False, default=0,
                                   help='Offset should be int.', location='args')
        self.reqparse.add_argument('sortBy', type=str, required=False, default="asc",
                                   help='SortBy should be str.', location='args')

    def get(self):
        args = self.reqparse.parse_args()
        limit = args["limit"]
        offset = args["offset"]
        sort_by = args["sortBy"]
        segment_name = args["segmentName"]
        response = {
            "segmentName": segment_name,
            "type": dataset_type,
            "urls": [],
        }
        try:
            segment = dataset.get_segment_by_name(segment_name)
        except KeyError:
            abort(404, f"Segment:'{segment_name}' does not exist.")

        if not dataset_type:
            for data in segment[offset: offset + limit]:
                if isinstance(data, Data):
                    response_data = {
                        "remotePath": data.target_remote_path,
                        "url": data.path
                    }
                else:
                    response_data = {
                        "remotePath": data.path,
                        "url": data.get_url()
                    }
                response["urls"].append(response_data)
        else:
            for frame in segment[offset: offset + limit]:
                response_frame = {
                    "frameId": frame.frame_id if hasattr(frame, "frame_id") else "",
                    "frame": []
                }
                for sensor_name, data in frame.items():
                    response_data = {
                        "sensor_name": sensor_name,
                        "timestamp": data.timestamp if hasattr(data, "timestamp") else 0
                    }
                    if isinstance(data, Data):
                        response_data["remotePath"] = data.target_remote_path
                        response_data["url"] = data.path
                    else:
                        response_data["remotePath"] = data.path
                        response_data["url"] = data.get_url()
                    response_frame["frame"].append(response_data)
                response["urls"].append(response_frame)

        response["offset"] = offset
        response["recordSize"] = len(response["urls"])
        response["totalCount"] = len(segment)

        if sort_by == "desc":
            response["urls"].reverse()
        return response


Label_Types = [
    {
        "labelKey": "BOX2D",
        "labelType": "2D BOX"
    },
    {
        "labelKey": "CLASSIFICATION",
        "labelType": "CLASSIFICATION"
    },
    {
        "labelKey": "POLYGON2D",
        "labelType": "2D POLYGON"
    },
    {
        "labelKey": "POLYLINE2D",
        "labelType": "2D POLYLINE"
    },
    {
        "labelKey": "CUBOID2D",
        "labelType": "2D CUBOID"
    },
    {
        "labelKey": "BOX3D",
        "labelType": "3D BOX"
    },
    {
        "labelKey": "KEYPOINTS2D",
        "labelType": "KEYPOINTS"
    },
    {
        "labelKey": "SENTENCE",
        "labelType": "Audio Sentence"
    },
]


class LabelTypeList(Resource):
    def get(self):
        return {"labelTypes":  Label_Types}


class LabelList(Resource):
    def __init__(self):
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument('segmentName', type=str, required=False, default="",
                                   help='SegmentName should be str.', location='args')
        self.reqparse.add_argument('limit', type=int, required=False, default=128,
                                   help='Limit should be int.', location='args')
        self.reqparse.add_argument('offset', type=int, required=False, default=0,
                                   help='Offset should be int.', location='args')
        self.reqparse.add_argument('sortBy', type=str, required=False, default="asc",
                                   help='SortBy should be str.', location='args')

    def get(self):
        args = self.reqparse.parse_args()
        limit = args["limit"]
        offset = args["offset"]
        sort_by = args["sortBy"]
        segment_name = args["segmentName"]
        response = {
            "segmentName": segment_name,
            "type": dataset_type,
            "labels": []
        }
        try:
            segment = dataset.get_segment_by_name(segment_name)
        except KeyError:
            abort(404, f"Segment:'{segment_name}' does not exist.")

        if not dataset_type:
            for data in segment[offset: offset + limit]:
                response_data = {
                    "remotePath": data.target_remote_path if isinstance(data, Data) else data.path,
                    "label": data.label.dumps()
                }
                response["labels"].append(response_data)

        else:
            for frame in segment[offset: offset + limit]:
                response_frame = {
                    "frameId": frame.frame_id if hasattr(frame, "frame_id") else "",
                    "frame": []
                }
                for sensor_name, data in frame.items():
                    response_data = {
                        "sensor_name": sensor_name,
                        "remotePath": data.target_remote_path if isinstance(data, Data) else data.path,
                        "timestamp": data.timestamp if hasattr(data, "timestamp") else 0,
                        "label": data.label.dumps()
                    }
                    response_frame["frame"].append(response_data)
                response["labels"].append(response_frame)

        response["offset"] = offset
        response["recordSize"] = len(response["labels"])
        response["totalCount"] = len(segment)

        if sort_by == "desc":
            response["labels"].reverse()
        return response


class SensorList(Resource):
    def __init__(self):
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument('segmentName', type=str, required=False, default="",
                                   help='SegmentName should be str.', location='args')

    def get(self):
        if not dataset_type:
            abort(404, "Please give fusion dataset.")

        args = self.reqparse.parse_args()
        segment_name = args["segmentName"]

        try:
            segment = dataset.get_segment_by_name(segment_name)
        except KeyError:
            abort(404, f"Segment:'{segment_name}' does not exist.")

        return {
            "segmentName": segment_name,
            "sensors": segment.sensors.dumps()
        }


api.add_resource(SegmentList, '/segments')
api.add_resource(Catalog, '/catalogs')
api.add_resource(LabelList, '/labels')
api.add_resource(SensorList, '/sensors')
api.add_resource(DataUriList, '/data/urls')
api.add_resource(LabelTypeList, '/labelTypes')


if __name__ == '__main__':
    app.run(debug=True)
