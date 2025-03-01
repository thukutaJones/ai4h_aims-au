import argparse
import os
import re
import string
import subprocess
import time

import pandas as pd
import requests
import torch
import torchmetrics
from tqdm import tqdm

# This is how many words of context (left/right) to take.
CONTEXT_SIZE = 50
LLAMA_SERVER_URL = "http://localhost:8080/v1/chat/completions"

class_names = [
    "approval",
    "signature",
    "criterion1",
    "criterion2_structure",
    "criterion2_operations",
    "criterion2_supplychains",
    "criterion3_risks",
    "criterion4_mitigation",
    "criterion4_remediation",
    "criterion5_assessment",
    "criterion6_consultation",
]


def fix_column_names(df):
    # Rename the column names to be consistent with AU
    if "sentence_statement_id" in df.columns and "statement_id" not in df.columns:
        df.rename(columns={"sentence_statement_id": "statement_id"}, inplace=True)
    if "sentence_orig_idxs" in df.columns and "sentence_id" not in df.columns:
        df.rename(columns={"sentence_orig_idxs": "sentence_id"}, inplace=True)
    if "sentence_orig_text" in df.columns and "sentence_text" not in df.columns:
        df.rename(columns={"sentence_orig_text": "sentence_text"}, inplace=True)
    if "sentence" in df.columns and "sentence_text" not in df.columns:
        df.rename(columns={"sentence": "sentence_text"}, inplace=True)

    # Modify the 'sentence_sentence_statement_id' column to remove the 'SID_' prefix
    # Check if statement_id is string and contains 'SID_'
    if isinstance(df["statement_id"], str) and df["statement_id"].str.contains("SID_").any():
        df["statement_id"] = df["statement_id"].str.replace("SID_", "", regex=False)
    return df


def generate_context_inputs(df):
    # Group by sentence_statement_id so we can handle each statement independently.
    grouped = df.groupby("statement_id", group_keys=False)

    all_outputs = []

    for sentence_statement_id, subdf in grouped:
        # Sort the rows in the correct reading order (by sentence_orig_idxs, presumably ascending):
        subdf_sorted = subdf.sort_values("sentence_id").reset_index(drop=True)

        # 1. Build one flat list of all words for this statement
        all_words = []
        offsets = []  # will store (start_offset, end_offset) for each row’s sentence
        current_offset = 0

        for text in subdf_sorted["sentence_text"]:
            words = text.split()
            start_offset = current_offset
            end_offset = start_offset + len(words)
            offsets.append((start_offset, end_offset))
            all_words.extend(words)
            current_offset = end_offset

        # 2. Now assign context to each row
        for i in range(len(subdf_sorted)):
            start_off, end_off = offsets[i]

            # Clip the 50 words before/after to the statement boundaries
            context_start = max(0, start_off - CONTEXT_SIZE)
            context_end = min(len(all_words), end_off + CONTEXT_SIZE)

            context_words = all_words[context_start:context_end]
            text_with_context = " ".join(context_words)

            # Store it in a new column:
            subdf_sorted.loc[i, "text_with_context"] = text_with_context

        # After we’ve built this for every row in subdf_sorted, collect results
        all_outputs.append(subdf_sorted)

    # 3. Concatenate all statement sub-dataframes into one final DF
    df_with_context = pd.concat(all_outputs, ignore_index=True)

    # (Optional) Save to a CSV to inspect
    df_with_context.to_csv("output_with_context.csv", index=False)


def get_prompt(class_of_interest_index):
    class_specific_prompts = [
        """You are currently looking for sentences in statements that describe whether the statement is approved by the principal governing body of one or more reporting entities. This governing body holds the primary responsibility for governance of the reporting entity. The Act explicitly prohibits the delegation of this approval authority to individuals, executive committees, sub-committees, or workgroups. It is crucial for entities to clearly identify that the approval has come directly from this sole governing body without ambiguity. Terms like “Executive Leadership Committee,” “Members of the Board,” or “Senior Executive on behalf of the Board” do not sufficiently demonstrate that the principal governing body itself has approved the modern slavery statement. Additionally, descriptions such as "considered by the board" or "put forward to the board" are also inadequate to fulfill the requirement of direct approval by the governing body.
    If there is only a single reporting entity, the approval must come from its sole principal governing body.
    Otherwise, for joint statements made by multiple reporting entities, there are three options for the approval of a statements:
    The principal governing body of each reporting entity covered by the statement approves the statement.
    The principal governing body of a higher entity (such as a global parent), which is in a position to influence or control each entity covered by the statement, approves the statement. The higher entity does not have to be a reporting entity itself.
    If it is not practicable to comply with other options, the principal governing body of at least one reporting entity covered by the statement may approve the statement. In this case, the statement must also explain why this option was taken.
    We therefore consider any sentence containing language that provides such approval as relevant.
    """,
        """You are currently looking for sentences in statements that describe whether the statement is signed by a responsible member of the reporting entity. A responsible member is a decision-making member of the reporting entity. Where a reporting entity has a principal governing body, the responsible member will be a member of that body. A statement cannot be signed by individuals who do not meet the definition of a responsible member. It is important to ensure that the text accompanying the signature shows that the individual signing has the authority to sign the statement by including all the following information: the name of the responsible member; the title or position (e.g., CEO, Director, Member of the Board, Chairman, President, etc.) of the responsible member; and a copy of their signature (an image) OR wording to explicitly indicate the signature of the statement (e.g. “signed by”, “signature”, “s/”, a signature ID, or any form of electronic signature).""",
        """ """,
        """You are currently looking for sentences in statements that describe the structure of an entity, where structure refers to the legal and organizational form of the reporting entity. A reporting entity can describe its structure in multiple ways. For example, it can explain its general structure, which covers the entity’s legal classification (company, partnership, trust, etc.). If the entity is part of a larger group, the statement should provide details on the general structure of the overall group (upstream and downstream from the reporting entity). The statement may also describe the approximate number of workers employed by the entity and the registered office location of the entity (i.e., its statutory address). Additionally, it can explain what the entity owns or controls, such as a foreign subsidiary. Identifying any trading names or brand names associated with the reporting entity and entities it owns or controls is also relevant. Relevant examples include descriptions of “corporate structure,” “governance structure,” the company type, parent group details, number of employees, owned brand names, and the registered office location. Irrelevant examples include isolated information about the company name or ABN, establishment year, or vague statements such as “We operate in Australia”, “We make this statement on behalf of our associated entities.” ,“This statement covers our wholly-owned subsidiaries.” or “We work in the XYZ sector with offices located in 123 countries.”. """,
        """You are currently looking for sentences in statements that describe the operations of an entity, where operations refer to the activities undertaken by the reporting entity or any of its owned or controlled entities to pursue its business objectives and strategy in Australia or overseas. The description of operations can include explaining the nature and types of activities undertaken by the entity (e.g., mining, retail, manufacturing) and any entities it owns or controls, identifying the countries or regions where the entity’s operations are located or take place, and providing facts and figures about the entity’s operations, such as factories, offices, franchises, and/or stores. Examples of relevant information include descriptions of employment activities, such as operating manufacturing facilities; processing, production, R&D, or construction activities; charitable or religious activities; the purchase, marketing, sale, delivery, or distribution of products or services; ancillary activities required for the main operations of the entity; financial investments, including internally managed portfolios and assets; managed/operated joint ventures; and leasing of property, products, or services. However, only the operations of the reporting entity and its owned/controlled entities are relevant, while the description of operations of parent or partner companies is not considered relevant. """,
        """You are currently looking for sentences in statements that describe the SUPPLY
CHAINS of an entity, where supply chains refer to the sequences of processes involved in
the procurement of products and services (including labour) that contribute to the reporting
entity’s own products and services. The description of a supply chain can be related, for
example, to 1) the products that are provided by suppliers; 2) the services provided by
suppliers, or 3) the location, category, contractual arrangement, or other attributes that
describe the suppliers. Any sentence that contains these kinds of information is considered
relevant. Descriptions that apply to indirect suppliers (i.e. suppliers-of-suppliers) are
considered relevant. Descriptions of the supply chains of entities owned or controlled by the
reporting entity making the statement are also considered relevant. However, descriptions of
’downstream’ supply chains, i.e. of how customers and clients of the reporting entity use its
products or services, are NOT considered relevant. Finally, sentences that describe how the
reporting entity lacks information on some of its supply chain, or how some of its supply
chains are still unmapped or unidentified, are also considered relevant.
""",
        """You are currently looking for sentences in statements that describe the modern slavery risks identified by reporting entities. Specifically, these statements should detail risks in the entities' operations and supply chains that have the potential to cause or contribute to modern slavery. Descriptions of other business risks such as health risks (due to hazardous environments or materials), regulatory risks, or environmental risks are not considered relevant themselves. For a sentence to be considered relevant, it only needs to identify or describe a modern slavery risk; it does not need to describe the process used to identify, assess, or address that risk. A reporting entity might describe its modern slavery risks linked to specific areas, such as geographic regions (e.g., Indonesia), industries (e.g., electronics), commodities (e.g., palm oil), workforces (e.g., migrant workers), or global events (e.g., COVID-19). The description of any past or current instances of modern slavery within the entity's operations or supply chains is also considered relevant information. It is insufficient to present a hypothetical case of modern slavery or state that the entire reporting entity has a “low risk” of modern slavery or that its operations or supply chains are “slavery-free”; the reporting entity needs to be specific.  Other examples of irrelevant or vague sentences include “Modern slavery has the potential to exist in the technology sector.” or “Modern slavery is defined as forced labor, debt bondage, or servitude.” In both cases, the link to the entity’s operations or supply chains is missing. """,
        """After describing the modern slavery risks they face, reporting entities are required to describe how these risks are identified, assessed, and/or mitigated. You are currently looking for descriptions of RELEVANT ACTIONS to identify, assess, or mitigate modern slavery risks.
Examples of RELEVANT ACTIONS may include:
Participating in or launching industry initiatives (e.g. UN Global Compact, or Finance Against Slavery and Trafficking Initiative), or providing financial support to such initiatives;
Requiring suppliers and third-party entities to comply and follow with internal, regional, or international policies or labor laws (e.g. by adopting a supplier code of conduct or contract terms);
Ensuring that employees, suppliers, or other stakeholders are aware of the company’s policies and requirements (e.g. by conducting training sessions on modern slavery or on policies);
Requiring employees to follow company guidelines regarding human rights, labour rights, association (or unionization) rights, responsible recruitment, and/or responsible procurement (e.g. by adopting a code of conduct or a code of ethics that is linked with modern slavery);
Adopting, drafting, updating, or upholding a policy, framework, standard, or program that is meant to identify risks of modern slavery; for instance, a company may propose a formal process for:
Conducting desk research by reviewing secondary sources of information (e.g. reports from international organizations, frameworks such as the UN Guiding Principles, or research papers and articles on modern slavery) or by reviewing external expert advisers’ inputs;
Utilizing risk management tools or software (e.g. SEDEX);
Conducting risk-based questionnaires, surveys, or interviews with employees, suppliers, or other stakeholders;
Auditing, screening, or directly engaging with employees, suppliers, or other stakeholders.
Committing to paying all employees, migrant workers, temporary workers, and third-party workers a “living wage”, to support their freedom of association (or unionization), or to support collective bargaining;
Adopting a code of conduct or code of ethics that relates to combatting or preventing modern slavery and forced labour practices;
Incorporating provisions for onboarding and engaging with suppliers, including contract clauses and requirements, and extending the code of conduct to cover suppliers;
Enforcing responsible recruitment of workers (e.g., by not allowing the payment of recruitment fees by workers, or by not withholding part of their compensation for housing or licensing costs);
Implementing responsible procurement practices when establishing new supply chains (e.g., no excessive pressure on lead times for products or services they source);
Having a zero-tolerance policy that would force the entity to take action regarding threats, regarding intimidation and attacks against human rights defenders, or regarding modern slavery cases tied to their suppliers;
Establishing a whistleblowing policy that encourages and safeguards workers, employees, suppliers, or other stakeholders when reporting concerns;
Ensuring that modern slavery cases or concerns are reported and investigated (e.g. by adopting a whistleblowing hotline or other reporting mechanism);
Ensuring the application of these policies or frameworks by having an executive or board member participate in their elaboration or in oversight committees.
Notes:
Vague language stating that the reporting entity has “zero tolerance” towards modern slavery without mentioning a policy in this regard,  or that it “is committed to fight” modern slavery is very common. However, such declarations do not describe a RELEVANT ACTION unless they are  linked for example with a company policy.
Not all policies adopted by an entity may be related to modern slavery. If a policy does not explicitly cover modern slavery, then the statement should provide a justification why adopting this policy should be considered a relevant mitigation action.
Examples of irrelevant text:
“Company XYZ recognizes the importance of acting to reduce the risk of modern slavery in its operations and supply chains”, or “We are committed to maintaining the highest level of integrity and honesty throughout all aspects of our business.” (there is no description of a real action here)
“Our approach towards modern slavery is zero-tolerance.” (unless they have a policy or enforce this approach somehow, this is too vague)
“We respect human rights and expect the same from our suppliers”, or “We expect our suppliers to respect local laws.” (the “expectation” is too weak since there is no description of a requirement or enforcement action towards suppliers; this would be relevant if this was for example enforced through a supplier code-of-conduct or through contract terms)
“We have adopted a new sustainability policy” or “We have adopted a new worker safety policy.” (irrelevant unless they clearly state that their definition of “sustainability” or “worker safety” covers modern slavery elsewhere in the statement — same reasoning applies for example to “worker health”, “diversity”, ”discrimination”, and “harassment” policies)
“We follow the law.” (irrelevant, as this is always expected of the reporting entities)
Any description of audit measures or other measures meant to assess the effectiveness of existing PROCESSES and ACTIONS meant to prevent modern slavery (e.g. an assestment of a supplier audit process)
Any description of tools (e.g. dashboards, software) used to monitor and track modern slavery risks (only the development and use of those tools are actions and therefore relevant)""",
        """You are currently looking for sentences in statements that describe remediation actions for modern slavery cases. If one or more cases (or, synonymously, incidents, issues, or situations) have been declared by the reporting entity where it caused or contributed to modern slavery, the statement should describe the actions taken to remediate (or, synonymously, manage, address, or remedy) these cases. If no modern slavery case has been identified by the reporting entity, it may still describe actions used to remediate hypothetical cases if one should occur in the future. Relevant sentences are those that clearly refer to actions proposed by the reporting entity to remediate modern slavery cases, and not to actions proposed to mitigate the risks of modern slavery. Corrective actions and sanctions to remediate modern slavery cases include, for example: 1) conducting inquiries and investigations involving stakeholders, 2) providing worker assistance such as returning passports or helping file legal claims, 3) offering compensation for owed wages or penalties, 4) issuing formal apologies, 5) notifying management or authorities of incidents, and 6) ceasing business activities with non-compliant partners or suppliers by canceling or terminating contracts. Sentences declaring that "We understand the importance of workers knowing their rights and addressing violations when necessary" or "Remediation was not needed in this reporting period," are NOT considered relevant as they do not describe specific remediation actions.""",
        """You are currently looking for sentences in statements that describe how the reporting entity assesses the effectiveness of its actions taken to mitigate risks as well as to remediate modern slavery cases. Relevant sentences are those that describe actions to evaluate the success of implemented mitigation and remediation measures and processes in achieving their intended goals. Assessing the effectiveness of prior actions can be accomplished in many ways, for example, by examining or revising action plans or policies,  by a review committee or by conducting annual assessments. It can also be achieved by setting up internal checks or a feedback process on the effectiveness of actions between key departments (or across owned or controlled entities) to facilitate regular engagement and communication among procurement, human resource management, and legal teams. Other relevant assessment actions include engaging with industry groups, external auditors, trusted NGOs, and stakeholders, including the workers themselves, to provide feedback or conduct independent reviews of actions, as well as analyzing trends in reported modern slavery cases through grievance or whistleblowing mechanisms. Additionally, reporting and monitoring Key Performance Indicators (KPIs) to measure the effectiveness of actions is a relevant assessment measure. Examples of such KPIs include the number of contracts in which modern slavery cases were identified, the number of trainings delivered to employees, suppliers, or stakeholders, the awareness of employees and suppliers towards modern slavery as measured through polls, the number of initiatives taken to enhance suppliers' capacity to respond to modern slavery risks, the number of filed grievances, the resolution rates of reports filed through grievance or whistleblowing mechanisms, the number of completed audits, the number of visits to facilities operated by the entity or its subsidiaries, the number of assigned and completed corrective action plans, the number of contracts canceled due to identified modern slavery cases, and the number of identified modern slavery incidents.
It is important to note that entities often claim to have "improved their modern slavery risk mitigation" over a reporting period, but such statements are not relevant unless they detail how these improvements were observed and measured. Relevant text should explain how the entity determined the effectiveness of its actions. For example, if an entity states that it "improved its modern slavery risk mitigation actions following a comprehensive review process," this is relevant because it describes the process used to assess effectiveness. There is a clear distinction between actions taken to mitigate or remediate and those taken to assess the effectiveness of these mitigation or remediation actions. Assessing effectiveness involves a second “review” stage, which evaluates whether the prior actions achieved their goals. Sentences that are vague or do not explain how improvements were made or observed are not relevant. """,
        """ """,
    ]

    prompt = (
        """"<｜User｜>You are an analyst that inspects modern slavery declarations made by Australian reporting
    entities. You are specialized in the analysis of statements made with respect to the Australian
    Modern Slavery Act of 2018, and not of any other legislation. """
        + class_specific_prompts[class_of_interest_index]
        + """ Given the above definitions of what constitutes a relevant sentence, you will need to determine if a target sentence is relevant or not inside a larger block of text. The target sentence will first be provided by itself so you can know which sentence we want to classify. It will then be provided again as part of the larger block of text it originally came from (extracted from a PDF file) so you can analyze it with more context. While some of the surrounding sentences may be relevant according to the earlier definitions, we are only interested in classifying the target sentence according to the relevance of its own content. You must avoid labeling sentences with only vague descriptions or corporate talk (and no actual information) as relevant.

    The answer you provide regarding whether the sentence is relevant or not can only be 'YES' or 'NO', and nothing else.

    The target sentence to classify is the following:
    ------------
    {}
    ------------

    The same target sentence inside its original block of text:
    ------------
    {}
    ------------

    Question: Is the target sentence relevant? (YES/NO)<｜Assistant｜>"""
    )

    return prompt


session = requests.Session()  # Reuse connection


def run_deepseek(target_sentence, text, class_of_interest_index, debug=False):
    """Runs DeepSeek for a given prompt and returns the response."""
    prompt = get_prompt(class_of_interest_index)
    punct_remover = str.maketrans("", "", string.punctuation)

    no_regex = re.compile("no[^0-9a-zA-Z]")
    yes_regex = re.compile("yes[^0-9a-zA-Z]")
    current_prompt = prompt.format(target_sentence, text)
    max_tokens = 1000
    temperature = 0.6
    payload = {
        "messages": [{"role": "user", "content": current_prompt}],
        "n_predict": max_tokens,
        "temperature": temperature,
    }

    try:
        # Convert the payload dictionary to a JSON string
        response = session.post(LLAMA_SERVER_URL, json=payload)
        response.raise_for_status()
        result = response.json()["choices"][0]["message"]["content"]
        # Extract the 'content' field from the response (adjust if necessary)
        # result = response.json().get("content", "No response")
    except requests.exceptions.RequestException as e:
        print(f"Error querying Llama: {e}")
        result = None

    # Extract final answer
    final_answer_match = re.search(r"\s*(YES|NO)\s*$", result, re.IGNORECASE)

    final_answer = final_answer_match.group(1).strip() if final_answer_match else "Final Answer not found"

    # If the final answer is not found, then look for yes/no at the end of result
    if final_answer == "Final Answer not found":
        final_answer_match = re.search(r"(yes|no)$", result[:-20], re.IGNORECASE)
        final_answer = final_answer_match.group(1).strip() if final_answer_match else "Final Answer not found"

    # Normalize the answer
    normalized_answer = final_answer.lower().translate(punct_remover).strip()
    if normalized_answer == "no" or no_regex.match(normalized_answer):
        final_result = 0
    elif normalized_answer == "yes" or yes_regex.match(normalized_answer):
        final_result = 1
    else:
        final_result = -1
        if debug:
            print(f"GPT result '{result}' could not be processed (normalized: '{normalized_answer}')")

    if debug:
        print("Prompt: ", current_prompt)
        print("XXXXXXXXXXXXXXXXXXXXXXXXX")
        print("This is the model output:\n", result)
        print("XXXXXXXXXXXXXXXXXXXXXXXXX")
        print("Final result: ", final_result)
        print("\n\n")

    return final_result, result, normalized_answer


def main():
    """Minimal code to reload a fine-tuned model and perform inference on text stored in a pandas dataframe.
    The file has to be called by using custom args.
    For example: python classify_text_in_dataframe.py ckpt_path=checkpoint.ckpt experiment=config.yaml +input_csv_file=/a/file/path.csv +output_csv_file=/a/file/path.csv
    """
    # Add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("class_of_interest_index", type=int, help="Which class_of_interest_index to classify.")
    parser.add_argument("input_csv_file", type=str, help="Path to the input CSV file.")
    args = parser.parse_args()

    # Reshaping the annotated data to crete context for the target sentance
    df = pd.read_csv(args.input_csv_file)
    df = fix_column_names(df)
    generate_context_inputs(df)

    # Load the dataset
    csv_path = "output_with_context.csv"
    df = pd.read_csv(csv_path)

    targets = df["targets"].tolist()  # Extract the targets as a list

    # Manually specify the index for your class of interest
    class_of_interest_index = args.class_of_interest_index  # Replace with the index of your desired class

    # Retrieve the class name corresponding to the index
    class_of_interest = class_names[class_of_interest_index]

    # Output results
    # print(f"Targets: {targets}")
    print(f"Class names: {class_names}")
    print(f"Class index: {class_of_interest_index}")
    print(f"Class name for index {class_of_interest_index} is {class_of_interest}")

    # Initialize counters and storage

    count_sentence_was_annotated_with_class_of_interest = 0
    count_sentence_was_not_annotated_with_class_of_interest = 0
    data = []
    already_there = 0
    annotated = 0
    not_ok_results = 0
    not_ok_details = []
    tot_count = 0

    # Parameters
    max_amount = 9999  # Limit the computation
    print_count = True
    inter_query_wait_time_in_sec = 0.2  # Time to wait between queries
    debug = False

    # Load cached results if available
    prev_df = (
        pd.read_csv(f"XL_results/result_for_class_index_test{class_of_interest_index}.csv")
        if os.path.isfile(f"XL_results/result_for_class_index_test{class_of_interest_index}.csv")
        else None
    )  # Replace with pandas.read_csv if using a cached file
    predict_fun = run_deepseek  # Replace with the appropriate prediction function

    try:
        for i, row in tqdm(df.iterrows(), total=len(df)):
            tot_count += 1
            sentence_text = row["sentence_text"]
            sentence_statement_id = int(row["statement_id"])
            sentence_orig_idxs = int(row["sentence_id"])
            text_with_context = row["text_with_context"]

            # Check if the sentence is already cached
            if (
                prev_df is not None
                and (
                    (prev_df["sentence_statement_id"] == sentence_statement_id)
                    & (prev_df["sentence_orig_idxs"] == sentence_orig_idxs)
                ).any()
            ):
                # Retrieve cached result
                cached_row = prev_df[
                    (prev_df["sentence_statement_id"] == sentence_statement_id)
                    & (prev_df["sentence_orig_idxs"] == sentence_orig_idxs)
                ]
                if len(cached_row) != 1:
                    raise ValueError(f"Multiple cache rows found for: {sentence_statement_id}:{sentence_orig_idxs}")

                cached_row = cached_row.iloc[0]
                data.append(
                    [
                        sentence_statement_id,
                        sentence_orig_idxs,
                        cached_row.get("target_classes", []),
                        cached_row.get("predicted_classes", []),
                        cached_row.get("reasoning", ""),
                        cached_row.get("answer", ""),
                    ]
                )
                already_there += 1

            else:
                # Process new sentence
                target_classes = [int(x) for x in eval(row["targets"])]  # Parse targets correctly
                predicted_classes = [-1] * len(target_classes)  # Initialize predictions
                reasoning = ""
                answer = ""

                if target_classes[class_of_interest_index] > -1:
                    # Predict classification
                    predicted_class, reasoning, answer = predict_fun(
                        sentence_text, text_with_context, class_of_interest_index, debug=debug
                    )
                    time.sleep(inter_query_wait_time_in_sec)

                    predicted_classes[class_of_interest_index] = predicted_class
                    count_sentence_was_annotated_with_class_of_interest += 1

                    # Handle problematic results
                    if predicted_class == -1:
                        not_ok_results += 1
                        not_ok_details.append([sentence_statement_id, sentence_orig_idxs, reasoning, answer])

                else:
                    count_sentence_was_not_annotated_with_class_of_interest += 1

                # Append processed row
                data.append(
                    [sentence_statement_id, sentence_orig_idxs, target_classes, predicted_classes, reasoning, answer]
                )
                annotated += 1

            # Stop processing if the max limit is reached
            if max_amount > -1 and tot_count >= max_amount:
                print(f"Reached the max amount of {max_amount}")
                break

            # Print progress
            if print_count:
                print(f"Processed {tot_count} / {max_amount}")

            if i % 10 == 0:
                result_df = pd.DataFrame(
                    data,
                    columns=[
                        "sentence_statement_id",
                        "sentence_orig_idxs",
                        "target_classes",
                        "predicted_classes",
                        "reasoning",
                        "answer",
                    ],
                )
                result_df.to_csv(f"XL_results/result_for_class_index_test{class_of_interest_index}.csv", index=False)

                # Print summary
                print(
                    f"{already_there} sentences retrieved from cache. {annotated} sentences were newly annotated.\n"
                    f"Of these, {count_sentence_was_annotated_with_class_of_interest} were annotated with the class of interest, "
                    f"and {count_sentence_was_not_annotated_with_class_of_interest} were not.\n"
                    f"Total processed: {count_sentence_was_annotated_with_class_of_interest + count_sentence_was_not_annotated_with_class_of_interest}\n"
                    f"Problematic results: {not_ok_results}"
                )

                # Save problematic results
                if not_ok_details:
                    not_ok_df = pd.DataFrame(
                        not_ok_details, columns=["statement_id", "sentence_orig_idxs", "reasoning", "answer"]
                    )
                    not_ok_df.to_csv(f"XL_results/not_ok_for_class_index_{class_of_interest_index}.csv", index=False)

    finally:
        # Save results to CSV
        result_df = pd.DataFrame(
            data,
            columns=[
                "sentence_statement_id",
                "sentence_orig_idxs",
                "target_classes",
                "predicted_classes",
                "reasoning",
                "answer",
            ],
        )
        result_df.to_csv(f"XL_results/result_for_class_index_test{class_of_interest_index}.csv", index=False)

        # Print summary
        print(
            f"{already_there} sentences retrieved from cache. {annotated} sentences were newly annotated.\n"
            f"Of these, {count_sentence_was_annotated_with_class_of_interest} were annotated with the class of interest, "
            f"and {count_sentence_was_not_annotated_with_class_of_interest} were not.\n"
            f"Total processed: {count_sentence_was_annotated_with_class_of_interest + count_sentence_was_not_annotated_with_class_of_interest}\n"
            f"Problematic results: {not_ok_results}"
        )

        # Save problematic results
        if not_ok_details:
            not_ok_df = pd.DataFrame(
                not_ok_details, columns=["statement_id", "sentence_orig_idxs", "reasoning", "answer"]
            )
            not_ok_df.to_csv(f"XL_results/not_ok_for_class_index_{class_of_interest_index}.csv", index=False)

    print(f"Data Length: {len(data)}")
    print(f"Example Row Lengths: {[len(row) for row in data[:5]]}")


if __name__ == "__main__":
    main()
