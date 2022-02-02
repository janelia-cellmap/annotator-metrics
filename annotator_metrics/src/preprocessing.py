import os


def get_crops():
    directories = [
        "/groups/cosem/cosem/annotations/training/",
        "/groups/cosem/cosem/annotation_and_analytics/training/rymert/",
        "/groups/cosem/cosem/annotation_and_analytics/training/forknalln/",
        "/groups/cosem/cosem/annotation_and_analytics/training/ludwigh/",
    ]

    for group in [1]:
        for crop in ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]:
            output_dir = f"/groups/cosem/cosem/ackermand/annotation_and_analytics/group{group}-labels/group{group}_{crop}/"
            os.system(f"mkdir -p {output_dir}")
            for directory in directories:
                if directory == "/groups/cosem/cosem/annotations/training/":
                    group_crop_dir = (
                        f"{directory}/group{group}-labels/group{group}_{crop}/"
                    )
                else:
                    group_crop_dir = f"{directory}/group{group}-labels/"

                annotator_dirs = [
                    os.path.join(group_crop_dir, dI)
                    for dI in os.listdir(group_crop_dir)
                    if os.path.isdir(os.path.join(group_crop_dir, dI))
                ]

                for annotator_dir in annotator_dirs:
                    if f"group{group}_{crop}" in annotator_dir:
                        # print(f"ln -s {annotator_dir} {output_dir}")
                        os.system(f"ln -s {annotator_dir} {output_dir}")
