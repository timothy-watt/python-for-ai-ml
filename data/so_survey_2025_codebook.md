# SO 2025 Developer Survey — Column Codebook

**Source:** Stack Overflow 2025 Annual Developer Survey
**URL:** https://survey.stackoverflow.co/datasets
**License:** Open Database License (ODbL)
**Curated subset:** 15,000 rows — English-language respondents with salary data

## Columns Used in This Book

| Column | Type | Description | Used In |
|--------|------|-------------|---------|
| ResponseId | Int | Unique respondent identifier | Ch 3 |
| ConvertedCompYearly | Float | Annual compensation in USD | Ch 3–7, 9 |
| DevType | String (multi) | Developer role(s), semicolon-separated | Ch 3, 6, 8 |
| LanguageHaveWorkedWith | String (multi) | Languages used, semicolon-separated | Ch 3, 4, 6 |
| LanguageWantToWorkWith | String (multi) | Languages wanted, semicolon-separated | Ch 8 |
| YearsCodePro | String/Int | Years of professional coding | Ch 3, 5, 6, 7 |
| YearsCode | String/Int | Total years coding (incl. hobbyist) | Ch 5, 6 |
| Country | String | Respondent country | Ch 3, 4, 5, 9 |
| EdLevel | String | Highest education level completed | Ch 5, 6 |
| Employment | String | Employment status | Ch 6 |
| RemoteWork | String | Remote / hybrid / in-person | Ch 4, 6 |
| OrgSize | String | Organization size | Ch 6 |
| Age | String | Age range (binned) | Ch 4, 9 |
| Gender | String | Gender identity | Ch 9 (bias audit) |
| AIToolCurrently | String (multi) | AI tools currently used | Ch 4, 8 |
| AIToolInterested | String (multi) | AI tools interested in | Ch 8 |
| AITrustTeammates | String | Trust in AI tool recommendations | Ch 8 |
| AIBen | String | Perceived benefit of AI tools | Ch 8 |

## Notes on Multi-Value Columns

Several columns use a semicolon-separated multi-value format:
```
Example row: DevType = "Back-end developer;Full-stack developer"
```
To work with these in Pandas:
```python
df['DevType'].str.split(';').explode().value_counts()
```
This pattern is covered in detail in Chapter 3, Section 3.3.
