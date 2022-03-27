#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting thhe result to a new artifact
"""
import argparse
import logging

import numpy as np
import pandas as pd
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    run = wandb.init(project="nyc_airbnb", group="basic_data_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    logging.info("Downloading data from W&B")
    local_path = wandb.use_artifact(args.input_artifact).file()
    df = pd.read_csv(local_path)

    log_msg = f"Removing properties with price outside range [{args.min_price}, {args.max_price}]"
    logging.info(log_msg)
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()

    logging.info("Updating max value of minimum_nights")
    df["minimum_nights"] = np.where(df["minimum_nights"] > args.max_nights,
                                    args.max_nights,
                                    df['minimum_nights'])

    logging.info("Updating max value of number_of_reviews")
    df["number_of_reviews"] = np.where(df["number_of_reviews"] > args.max_number_reviews, 
                                       args.max_number_reviews, 
                                       df['number_of_reviews'])

    logging.info("Updating max value of reviews_per_month")
    df["reviews_per_month"] = np.where(df["reviews_per_month"] > args.max_reviews_per_month, 
                                       args.max_reviews_per_month, 
                                       df['reviews_per_month'])

    df.to_csv(args.output_artifact, index=None)

    logging.info("Creating and logging artifact with data clean to W&B")
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )

    artifact.add_file(args.output_artifact)
    run.log_artifact(artifact)
    logging.info("Artifact successfully saved to W&B")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Input data used for data cleaning",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Output of the data cleaning process",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Type of the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="Description of the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="Minimun price a property can have, in our model",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="Maximum price a property can have, in our model",
        required=True
    )

    parser.add_argument(
        "--max_nights", 
        type=int,
        help="Maximum number of nights someone needs to rent",
        required=True
    )

    parser.add_argument(
        "--max_number_reviews", 
        type=int,
        help="Maximum number of reviews a property can have,in our model",
        required=True
    )

    parser.add_argument(
        "--max_reviews_per_month", 
        type=float,
        help="Maximum number of reviews per month a porperty can have, in our model",
        required=True
    )


    args = parser.parse_args()

    go(args)
