#!/groups/scicompsoft/home/ackermand/miniconda3/envs/annotator-metrics/bin/python
import os
import re
import time
import smtplib
from email.mime.text import MIMEText
import datetime
import glob

import yaml
from yaml.loader import SafeLoader

import json


class Monitor:
    def __init__(self):
        self.base_dir = (
            "/groups/cellmap/cellmap/annotation_and_analytics/analysis/submission"
        )
        self.previous_submission_time = int(
            os.popen(f"tail -n 1 {self.base_dir}/submission_times.txt").read()
        )
        self.current_time = int(time.time())
        self.groups_to_info_dict = {}

        with open(
            "/groups/scicompsoft/home/ackermand/Programming/annotator-metrics/email_info.json"
        ) as f:
            email_info = json.load(f)
        self.gmail_user = email_info["gmail_user"]
        self.gmail_application_password = email_info["gmail_application_password"]
        self.recipients = email_info["recipients"]

    def get_modified_file_info(self):
        seconds_since_submission = self.current_time - self.previous_submission_time

        annotations_added_since_last_submission = []
        for annotation_dir_to_check in [
            "/groups/cellmap/cellmap/annotation_and_analytics/training/",
            "/groups/cellmap/cellmap/annotation_and_analytics/supplemental_variability_crops/variability_crops_personal",
        ]:
            current_annotations_added_since_last_submssion = os.popen(
                f'find "{annotation_dir_to_check}" -type f -name "*.tif" -newermt -"{seconds_since_submission} seconds"'
            ).read()
            annotations_added_since_last_submission += current_annotations_added_since_last_submssion.split(
                "\n"
            )

        non_annotator_dirs = [
            "paintera templates",
            "conversion_scripts",
            "textfile_templates",
        ]
        for annotation in annotations_added_since_last_submission:
            if (
                all(
                    [
                        non_annotator_dir not in annotation
                        for non_annotator_dir in non_annotator_dirs
                    ]
                )
                and annotation
            ):
                if "supplemental_variability_crops" not in annotation:
                    user_and_group = (
                        annotation.split("-labels")[0].split("training/")[-1].split("/")
                    )
                    user = user_and_group[0]
                    group = user_and_group[1]
                    if re.search(
                        f"{group}-labels/{group}_.*/{group}_*_.*\.tif", annotation
                    ):  # then is likely an actual annotation
                        if group not in self.groups_to_info_dict:
                            self.groups_to_info_dict[group] = {user: [annotation]}
                        else:
                            if user not in self.groups_to_info_dict[group]:
                                self.groups_to_info_dict[group][user] = [annotation]
                            else:
                                self.groups_to_info_dict[group][user].append(annotation)
                else:
                    user_and_group = (
                        annotation.split("group.*_crop")[0]
                        .split("variability_crops_personal/")[-1]
                        .split("/")
                    )
                    user = user_and_group[0]
                    group = user_and_group[1]
                    if re.search(
                        f"{group}/{group}_crop.*/{group}_*_.*\.tif", annotation
                    ):  # then is likely an actual annotation
                        if group not in self.groups_to_info_dict:
                            self.groups_to_info_dict[group] = {user: [annotation]}
                        else:
                            if user not in self.groups_to_info_dict[group]:
                                self.groups_to_info_dict[group][user] = [annotation]
                            else:
                                self.groups_to_info_dict[group][user].append(annotation)

    def submit_jobs(self):
        groups = self.groups_to_info_dict.keys()
        if groups:
            # Then we need to submit
            groups = ",".join(groups)
            os.system(
                f"echo {self.current_time} >> {self.base_dir}/submission_times.txt"
            )

            self.current_time = int(time.time())
            new_config = (
                f"{self.base_dir}/submission_configs/lsf-config-{self.current_time}"
            )
            os.system(f"cp -r {self.base_dir}/lsf-config-template {new_config}")

            os.system(
                f"sed -i 's/template_group/\"{groups}\"/g' {new_config}/run-config.yaml"
            )
            os.system(f"bsub -n 2 -P cellmap annotator-metrics {new_config} -n 48")
            monitor_new_annotations.send_email("submitted", new_config)

    def monitor_job_status(self):
        with open(f"{self.base_dir}/completed_submissions.txt") as f:
            completed_submissions = f.read()

        submission_configs = glob.glob(
            f"{self.base_dir}/submission_configs/lsf-config-*-*"
        )
        for submission_config in submission_configs:
            if submission_config not in completed_submissions:
                with open(f"{submission_config}/output.log") as f:
                    log = f.read()
                    if "Calculations completed successfully!" in log:
                        self.send_email("completed", submission_config)
                        os.system(
                            f"echo {submission_config} >> {self.base_dir}/completed_submissions.txt"
                        )
                    elif "Calculations failed!" in log:
                        self.send_email("failed", submission_config)
                        os.system(
                            f"echo {submission_config} >> {self.base_dir}/completed_submissions.txt"
                        )

    def send_email(self, status, config_path):

        dt = datetime.datetime.fromtimestamp(self.previous_submission_time)
        if status == "submitted":
            dt = datetime.datetime.fromtimestamp(self.previous_submission_time)
            body = f"You are receiving this email because the following annotations have been added/modified since {dt}:\n"
            for group, users_dict in self.groups_to_info_dict.items():
                body += f"\n{group}:\n"
                for user, annotations in users_dict.items():
                    body += f"\t{user}:\n"
                    for annotation in annotations:
                        body += f"\t\t{annotation}\n"
            body += "\n"
            body += f"The job has been submitted to the cluster using the config located at {config_path}"
            msg = MIMEText(body)
            msg["Subject"] = "Annotator Metrics Analysis Submission"
        else:
            dt = datetime.datetime.fromtimestamp(int(config_path.split("-")[2]))
            with open(f"{config_path}/run-config.yaml") as f:
                groups = yaml.load(f, Loader=SafeLoader)["required_settings"][
                    "group"
                ].split(",")
            msg = MIMEText(
                f"""You are receiving this email because the following analysis submitted {dt} has {status} for {groups}: {config_path}"""
            )
            msg["Subject"] = f"Annotator Metrics Analysis {status.capitalize()}!"

        sender = self.gmail_user
        msg["From"] = sender
        msg["To"] = ", ".join(self.recipients)
        s = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        s.ehlo()
        s.login(self.gmail_user, self.gmail_application_password)
        s.sendmail(sender, self.recipients, msg.as_string())


if __name__ == "__main__":
    monitor_new_annotations = Monitor()
    monitor_new_annotations.get_modified_file_info()
    monitor_new_annotations.submit_jobs()
    monitor_new_annotations.monitor_job_status()

    # submit
    # copy lsf-config over...instead get datetime of submission directory in case submission might take some time
