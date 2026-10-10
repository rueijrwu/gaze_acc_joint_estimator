# Source snapshot context

This snapshot preserves the source files used for this attempt in their original repository-relative layout. The copied source files remain byte-for-byte and their hashes are recorded in the attempt's `provenance.json`.

Raw fixation data remain at the repository root in `data/fixations/fixation_intervals.json`; raw data are not copied into this nested snapshot. Archived source-document links to repository data therefore resolve from the original repository context, not from inside this snapshot directory.
