


def add_label_column(run_df, qrels_df=None):
    if qrels_df is not None:
        # here, check if passaging has happenned
        if "%p" in run_df["docno"][0]:
            # compute the original docno
            run_df["orig_docno"] = run_df.apply(lambda x: row["docno"].split("%p")[1], axis=1)
            # join with qrels using the original docno
            run_df = run_df.merge(qrels_df, left_on=["qid", "orig_docno"], right_on=["qid", "docno"], how="left")
            # drop the original docno
            run_df = run_df.drop(["orig_docno"], axis=1)
        else:
            run_df = run_df.merge(qrels_df, on=["qid", "docno"], how="left")
        run_df["label"] = run_df["label"].fillna(0)
    if "label" in run_df.columns:
        qids_with_relevant = run_df[run_df["label"] > 0][["qid"]].drop_duplicates()
        final_DF = run_df.merge(qids_with_relevant, on="qid")
        if len(final_DF) == 0:
            raise ValueError("No queries with relevant documents")
        final_DF["label"] = final_DF["label"].fillna(0)
    else:
        final_DF = run_df
    
    return final_DF
