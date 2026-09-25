# =========================================================
# PREPARE DATA
# =========================================================

prices["date"] = pd.to_datetime(
    prices["date"],
    errors="coerce"
)

# ---------------------------------------------------------
# ITEM LOOKUP
# ---------------------------------------------------------

# Keep only the columns we need
item_lookup = items[
    ["item_code", "item"]
].copy()

# Make lookup key the same type
prices["item_code"] = prices["item_code"].astype(str)
item_lookup["item_code"] = item_lookup["item_code"].astype(str)

# Remove duplicate item codes
item_lookup = item_lookup.drop_duplicates(
    "item_code"
)

# If an item column already exists in prices,
# remove it before merging
if "item" in prices.columns:
    prices = prices.drop(
        columns=["item"]
    )

# Merge item name
prices = prices.merge(
    item_lookup,
    on="item_code",
    how="left"
)


# ---------------------------------------------------------
# PREMISE LOOKUP
# ---------------------------------------------------------

premise_lookup = premises[
    [
        "premise_code",
        "state",
        "district",
        "premise_type"
    ]
].copy()

# Make lookup key the same type
prices["premise_code"] = prices[
    "premise_code"
].astype(str)

premise_lookup["premise_code"] = premise_lookup[
    "premise_code"
].astype(str)

# Remove duplicate premise codes
premise_lookup = premise_lookup.drop_duplicates(
    "premise_code"
)

# Remove existing columns before merging
for column in [
    "state",
    "district",
    "premise_type"
]:

    if column in prices.columns:

        prices = prices.drop(
            columns=[column]
        )

# Merge premise information
prices = prices.merge(
    premise_lookup,
    on="premise_code",
    how="left"
)


# ---------------------------------------------------------
# CHECK LOOKUP RESULTS
# ---------------------------------------------------------

if "item" not in prices.columns:

    st.error(
        "The item column could not be created after "
        "merging lookup_item.csv."
    )

    st.write(
        "Available columns:",
        list(prices.columns)
    )

    st.stop()


if "district" not in prices.columns:

    st.error(
        "The district column could not be created after "
        "merging lookup_premise.csv."
    )

    st.write(
        "Available columns:",
        list(prices.columns)
    )

    st.stop()


# Check whether the lookup actually matched
if prices["item"].notna().sum() == 0:

    st.error(
        "Item codes in latest_prices.csv could not "
        "be matched with lookup_item.csv."
    )

    st.stop()


if prices["district"].notna().sum() == 0:

    st.error(
        "Premise codes in latest_prices.csv could not "
        "be matched with lookup_premise.csv."
    )

    st.stop()
