-- Semantic Model SQL Views
-- Generated from semantic model
-- Dialect: tsql

-- Create semantic schema
IF NOT EXISTS (SELECT * FROM sys.schemas WHERE name = 'semantic')
    EXEC('CREATE SCHEMA semantic');
GO

-- Fact: ContentAttributeValue
-- Source: dbo.ContentAttributeValue
CREATE OR ALTER VIEW semantic.ContentAttributeValue AS
SELECT
    ID -- ID,
    ContractProductID -- ContractProductID,
    ContentAttributeDefinitionID -- ContentAttributeDefinitionID,
    ProductPartID -- ProductPartID,
    Value -- Value,
    SeqNo -- SeqNo,
    DMKey -- DMKey,
    DMTable -- DMTable,
    DMCharKey -- DMCharKey,
    AdvertisementID -- AdvertisementID is being used as a status/association flag: NULL = no advertisement link; non-NULL = linked to an Advertisement record. To identify rows that are 'advertised' use AdvertisementID IS NOT NULL. If you need to ensure the referenced advertisement is itself active, join or test the Advertisement table (for example: AdvertisementID IS NOT NULL AND EXISTS (SELECT 1 FROM dbo.Advertisement a WHERE a.AdvertisementID = dbo.ContentAttributeValue.AdvertisementID AND a.IsActive = 1 /* or a.IsDeleted = 0 / a.Status = 'Active' depending on schema */)). Verify referential integrity and the Advertisement table's lifecycle flags before treating non-NULL as fully 'active'.,
    QA -- QA,
    HeaderLabel -- HeaderLabel,
    -- Measure: RecordCount = COUNT(*),
    -- Measure: TotalSeqNo = SUM(SeqNo),
    -- Measure: AverageSeqNo = AVG(SeqNo),
    -- Measure: MinSeqNo = MIN(SeqNo),
    -- Measure: MaxSeqNo = MAX(SeqNo),
    -- Measure: CountSeqNo = COUNT(SeqNo),
    -- Measure: TotalDMKey = SUM(DMKey),
    -- Measure: AverageDMKey = AVG(DMKey),
    -- Measure: MinDMKey = MIN(DMKey),
    -- Measure: MaxDMKey = MAX(DMKey),
    -- Measure: DistinctDMKeyCount = COUNT(DISTINCT DMKey)
FROM dbo.ContentAttributeValue
WHERE
    AdvertisementID IS NULL
    -- Filter for active records only
GO

-- Fact: FreeTemplateAttributeValue
-- Source: dbo.FreeTemplateAttributeValue
CREATE OR ALTER VIEW semantic.FreeTemplateAttributeValue AS
SELECT
    ID -- ID,
    FreeTemplateID -- FreeTemplateID,
    ContentAttributeDefinitionID -- ContentAttributeDefinitionID,
    ProductPartID -- ProductPartID,
    Value -- Value,
    SeqNo -- SeqNo,
    DMKey -- DMKey,
    DMTable -- DMTable,
    DMCharKey -- DMCharKey,
    FreeTemplateAdvertisementID -- FreeTemplateAdvertisementID is a nullable foreign-key-style status indicator. NULL means 'no association/unassigned'; any non-NULL value denotes an association to a FreeTemplateAdvertisement record. To treat 'active' as records that are assigned, filter with IS NOT NULL. If your system uses a sentinel (e.g. 0) treat that as non-active as well: WHERE FreeTemplateAdvertisementID IS NOT NULL AND FreeTemplateAdvertisementID <> 0. To be stricter, join to the referenced FreeTemplateAdvertisement table and check that the referenced row itself is not deleted/inactive (for example, INNER JOIN dbo.FreeTemplateAdvertisement fta ON fta.ID = FreeTemplateAdvertisementID AND fta.IsActive = 1).,
    QA -- QA,
    HeaderLabel -- HeaderLabel,
    -- Measure: RecordCount = COUNT(*),
    -- Measure: DistinctDMKeyCount = COUNT(DISTINCT DMKey),
    -- Measure: SumSeqNo = SUM(SeqNo),
    -- Measure: AverageSeqNo = AVG(SeqNo),
    -- Measure: MinSeqNo = MIN(SeqNo),
    -- Measure: MaxSeqNo = MAX(SeqNo),
    -- Measure: SumDMKey = SUM(DMKey),
    -- Measure: AverageDMKey = AVG(DMKey),
    -- Measure: MinDMKey = MIN(DMKey),
    -- Measure: MaxDMKey = MAX(DMKey)
FROM dbo.FreeTemplateAttributeValue
WHERE
    FreeTemplateAdvertisementID IS NULL
    -- Filter for active records only
GO

-- Fact: TaskLog
-- Source: dbo.TaskLog
CREATE OR ALTER VIEW semantic.TaskLog AS
SELECT
    ID -- ID,
    TaskID -- TaskID,
    Timestamp -- Timestamp,
    Outcome -- Outcome,
    Text -- Text,
    DMKey -- DMKey,
    DMTable -- DMTable,
    DMCharKey -- DMCharKey,
    OutcomeDate -- OutcomeDate,
    -- Measure: TaskLogCount = COUNT(*),
    -- Measure: DistinctDMKeyCount = COUNT(DISTINCT DMKey),
    -- Measure: TotalDMKey = SUM(DMKey),
    -- Measure: AverageDMKey = AVG(DMKey),
    -- Measure: MinDMKey = MIN(DMKey),
    -- Measure: MaxDMKey = MAX(DMKey)
FROM dbo.TaskLog
GO

-- Fact: TaskAssignment
-- Source: dbo.TaskAssignment
CREATE OR ALTER VIEW semantic.TaskAssignment AS
SELECT
    ID -- ID,
    TaskID -- TaskID,
    RoleID -- RoleID,
    AssignmentType -- AssignmentType,
    UserID -- UserID,
    DMKey -- DMKey,
    DMTable -- DMTable,
    DMCharKey -- DMCharKey,
    -- Measure: AssignmentCount = COUNT(*),
    -- Measure: DistinctDMKeyCount = COUNT(DISTINCT DMKey),
    -- Measure: DistinctAssignmentTypeCount = COUNT(DISTINCT AssignmentType),
    -- Measure: AssignmentTypeSum = SUM(AssignmentType),
    -- Measure: AssignmentTypeAverage = AVG(AssignmentType),
    -- Measure: AssignmentTypeMin = MIN(AssignmentType),
    -- Measure: AssignmentTypeMax = MAX(AssignmentType),
    -- Measure: DMKeyMin = MIN(DMKey),
    -- Measure: DMKeyMax = MAX(DMKey)
FROM dbo.TaskAssignment
GO

-- Fact: TimeLogger
-- Source: dbo.TimeLogger
CREATE OR ALTER VIEW semantic.TimeLogger AS
SELECT
    ID -- ID,
    EventGroup -- EventGroup,
    EventDate -- EventDate,
    Descr -- Descr,
    RowCountVal -- RowCountVal,
    DataSize -- DataSize,
    LogSize -- LogSize,
    -- Measure: RecordCount = COUNT(ID),
    -- Measure: DistinctIDCount = COUNT(DISTINCT ID),
    -- Measure: MinID = MIN(ID),
    -- Measure: MaxID = MAX(ID),
    -- Measure: TotalRowCount = SUM(RowCountVal),
    -- Measure: AverageRowCount = AVG(RowCountVal),
    -- Measure: MinRowCount = MIN(RowCountVal),
    -- Measure: MaxRowCount = MAX(RowCountVal),
    -- Measure: TotalDataSize = SUM(DataSize),
    -- Measure: AverageDataSize = AVG(DataSize),
    -- Measure: MinDataSize = MIN(DataSize),
    -- Measure: MaxDataSize = MAX(DataSize),
    -- Measure: TotalLogSize = SUM(LogSize),
    -- Measure: AverageLogSize = AVG(LogSize),
    -- Measure: MinLogSize = MIN(LogSize),
    -- Measure: MaxLogSize = MAX(LogSize),
    -- Measure: AverageDataSizePerRow = CASE WHEN SUM(RowCountVal)=0 THEN NULL ELSE 1.0*SUM(DataSize)/SUM(RowCountVal) END
FROM dbo.TimeLogger
GO

-- Fact: TargetGroupItem
-- Source: dbo.TargetGroupItem
CREATE OR ALTER VIEW semantic.TargetGroupItem AS
SELECT
    ID -- ID,
    BusinessPointID -- BusinessPointID,
    CampaignID -- CampaignID,
    SourceCampaignID -- SourceCampaignID,
    CaseID -- CaseID,
    CustomerName -- CustomerName,
    Brandname -- Brandname,
    Prefecture -- Prefecture,
    Municipality -- Municipality,
    City -- City,
    StreetName -- StreetName,
    StreetNumber -- StreetNumber,
    NumberExtension -- NumberExtension,
    ZipCode -- ZipCode,
    Phone -- Phone,
    CustomerProfession -- CustomerProfession,
    ReliabilityStatus -- This NVARCHAR column holds a categorical reliability status using color labels. NULL means the status hasn't been provided or calculated. Non-NULL values are the assigned reliability categories; their exact business meanings (especially PURPLE) should be confirmed with stakeholders. Common alternative filters you might use depending on intent: to get only assessed records use ReliabilityStatus IS NOT NULL (shown above); to get only healthy/operational records use ReliabilityStatus = 'GREEN'; to get records that are not in a failed state exclude RED with ReliabilityStatus <> 'RED' AND ReliabilityStatus IS NOT NULL.,
    MainHeadingID -- MainHeadingID,
    MainHeadingCode -- MainHeadingCode,
    MainHeadingName -- MainHeadingName,
    Code -- Code,
    ProcessID -- ProcessID,
    DMKey -- DMKey,
    DMTable -- DMTable,
    DMCharKey -- DMCharKey,
    PaymentRefusal -- PaymentRefusal,
    Disappeared -- Disappeared is a nullable BIT that marks whether a TargetGroupItem has been removed/ceased to exist. A stored 1 should be interpreted as disappeared (inactive), and 0 as not disappeared (active). NULL typically means the flag wasn't set or the state is unknown; you should confirm with the data owner whether NULL means 'unknown' or implicitly 'not disappeared'. If you want to treat unknowns as active, filter with ISNULL(Disappeared, 0) = 0 (or WHERE Disappeared = 0 OR Disappeared IS NULL). If NULL should be excluded until clarified, use WHERE Disappeared = 0.,
    PrecariousnessLegal -- PrecariousnessLegal,
    ContinuousPaymentRefusal -- ContinuousPaymentRefusal,
    SealedCheck -- SealedCheck,
    -- Measure: RecordCount = COUNT(*),
    -- Measure: MainHeadingID_Count = COUNT(MainHeadingID),
    -- Measure: DMKey_Count = COUNT(DMKey),
    -- Measure: DistinctMainHeadingCount = COUNT(DISTINCT MainHeadingID),
    -- Measure: DistinctDMKeyCount = COUNT(DISTINCT DMKey),
    -- Measure: SumMainHeadingID = SUM(MainHeadingID),
    -- Measure: AverageMainHeadingID = AVG(MainHeadingID),
    -- Measure: MinMainHeadingID = MIN(MainHeadingID),
    -- Measure: MaxMainHeadingID = MAX(MainHeadingID),
    -- Measure: SumDMKey = SUM(DMKey),
    -- Measure: AverageDMKey = AVG(DMKey),
    -- Measure: MinDMKey = MIN(DMKey),
    -- Measure: MaxDMKey = MAX(DMKey)
FROM dbo.TargetGroupItem
WHERE
    Disappeared IS NULL
    -- Filter for active records only
GO

-- Fact: PricelistItemDetail
-- Source: dbo.PricelistItemDetail
CREATE OR ALTER VIEW semantic.PricelistItemDetail AS
SELECT
    ID -- ID,
    PricelistItemID -- PricelistItemID is an identifying foreign/key column, not a status indicator. It cannot be NULL (per schema). A populated value simply points to the related PricelistItem record. To determine whether a detail row is 'active' you must consult the status column on the referenced PricelistItem (or another dedicated status column), not this ID column.,
    ProductID -- ProductID,
    ProductPartID -- ProductPartID,
    KitID -- KitID,
    MeasurementType -- MeasurementType,
    UnitAmount -- UnitAmount,
    ComplexPrice -- ComplexPrice,
    DMKey -- DMKey,
    DMTable -- DMTable,
    DMCharKey -- DMCharKey,
    AllowableQuantities -- AllowableQuantities,
    AmountsPerAllowableQuantities -- AmountsPerAllowableQuantities,
    -- Measure: TotalUnitAmount = SUM(UnitAmount),
    -- Measure: AverageUnitAmount = AVG(UnitAmount),
    -- Measure: MinimumUnitAmount = MIN(UnitAmount),
    -- Measure: MaximumUnitAmount = MAX(UnitAmount),
    -- Measure: RecordCount = COUNT(DMKey),
    -- Measure: DistinctRecordCount = COUNT(DISTINCT DMKey),
    -- Measure: MeasurementTypeCount = COUNT(MeasurementType),
    -- Measure: DistinctMeasurementTypeCount = COUNT(DISTINCT MeasurementType),
    -- Measure: MinimumMeasurementType = MIN(MeasurementType),
    -- Measure: MaximumMeasurementType = MAX(MeasurementType)
FROM dbo.PricelistItemDetail
WHERE
    PricelistItemID IS NULL
    -- Filter for active records only
GO

-- Fact: AuditLog
-- Source: dbo.AuditLog
CREATE OR ALTER VIEW semantic.AuditLog AS
SELECT
    ID -- ID,
    UserID -- UserID,
    EntityID -- EntityID,
    EntityType -- EntityType,
    Timestamp -- Timestamp,
    LogInfo -- LogInfo,
    DMKey -- DMKey,
    DMTable -- DMTable,
    DMCharKey -- DMCharKey,
    -- Measure: RecordCount = COUNT(*),
    -- Measure: DistinctEntityCount = COUNT(DISTINCT EntityID),
    -- Measure: NonNullEntityCount = COUNT(EntityID),
    -- Measure: SumEntityID = SUM(EntityID),
    -- Measure: AverageEntityID = AVG(EntityID),
    -- Measure: MinEntityID = MIN(EntityID),
    -- Measure: MaxEntityID = MAX(EntityID),
    -- Measure: DistinctEntityTypeCount = COUNT(DISTINCT EntityType),
    -- Measure: NonNullEntityTypeCount = COUNT(EntityType),
    -- Measure: SumEntityType = SUM(EntityType),
    -- Measure: AverageEntityType = AVG(EntityType),
    -- Measure: MinEntityType = MIN(EntityType),
    -- Measure: MaxEntityType = MAX(EntityType),
    -- Measure: DistinctDMKeyCount = COUNT(DISTINCT DMKey),
    -- Measure: NonNullDMKeyCount = COUNT(DMKey),
    -- Measure: SumDMKey = SUM(DMKey),
    -- Measure: AverageDMKey = AVG(DMKey),
    -- Measure: MinDMKey = MIN(DMKey),
    -- Measure: MaxDMKey = MAX(DMKey)
FROM dbo.AuditLog
GO

-- Fact: BusinessPointComment
-- Source: dbo.BusinessPointComment
CREATE OR ALTER VIEW semantic.BusinessPointComment AS
SELECT
    ID -- ID,
    BusinessPointID -- BusinessPointID,
    UserID -- UserID,
    CommentDate -- CommentDate,
    CommentText -- CommentText,
    CommentType -- CommentType,
    -- Measure: TotalComments = COUNT(*),
    -- Measure: DistinctUsers = COUNT(DISTINCT UserID),
    -- Measure: DistinctBusinessPoints = COUNT(DISTINCT BusinessPointID),
    -- Measure: AverageCommentsPerUser = CAST(COUNT(*) AS FLOAT) / NULLIF(COUNT(DISTINCT UserID), 0),
    -- Measure: AverageCommentsPerBusinessPoint = CAST(COUNT(*) AS FLOAT) / NULLIF(COUNT(DISTINCT BusinessPointID), 0),
    -- Measure: MinCommentType = MIN(CommentType),
    -- Measure: MaxCommentType = MAX(CommentType),
    -- Measure: AverageCommentType = AVG(CAST(CommentType AS FLOAT))
FROM dbo.BusinessPointComment
GO

-- Fact: DialerTask
-- Source: dbo.DialerTask
CREATE OR ALTER VIEW semantic.DialerTask AS
SELECT
    ID -- ID,
    BusinessPointID -- BusinessPointID,
    SourceTaskID -- SourceTaskID,
    CurrentTaskID -- CurrentTaskID,
    PendingImport -- PendingImport,
    PermitCall -- PermitCall,
    DialerCampaignID -- DialerCampaignID,
    ImportDate -- ImportDate,
    RequiresSynchronization -- RequiresSynchronization,
    LastSynchDate -- LastSynchDate,
    TargetGroupItemValueID -- TargetGroupItemValueID,
    LastResubmitDate -- LastResubmitDate,
    LastResumeDate -- LastResumeDate,
    ExportDate -- ExportDate,
    BatchID -- BatchID,
    ProcessID -- ProcessID,
    CampaignID -- CampaignID,
    RecallDate -- RecallDate,
    NotFound -- NotFound,
    NotFoundDialerDate -- NotFoundDialerDate,
    NotFoundImportDate -- NotFoundImportDate,
    DialerGroupID -- DialerGroupID,
    -- Measure: TaskCount = COUNT([ID]),
    -- Measure: DistinctTaskIDs = COUNT(DISTINCT [ID]),
    -- Measure: MinTaskID = MIN([ID]),
    -- Measure: MaxTaskID = MAX([ID]),
    -- Measure: DistinctBusinessPointCount = COUNT(DISTINCT [BusinessPointID]),
    -- Measure: MinBusinessPointID = MIN([BusinessPointID]),
    -- Measure: MaxBusinessPointID = MAX([BusinessPointID]),
    -- Measure: DistinctSourceTaskCount = COUNT(DISTINCT [SourceTaskID]),
    -- Measure: MinSourceTaskID = MIN([SourceTaskID]),
    -- Measure: MaxSourceTaskID = MAX([SourceTaskID]),
    -- Measure: DistinctCurrentTaskCount = COUNT(DISTINCT [CurrentTaskID]),
    -- Measure: MinCurrentTaskID = MIN([CurrentTaskID]),
    -- Measure: MaxCurrentTaskID = MAX([CurrentTaskID]),
    -- Measure: DistinctDialerCampaignCount = COUNT(DISTINCT [DialerCampaignID]),
    -- Measure: MinDialerCampaignID = MIN([DialerCampaignID]),
    -- Measure: MaxDialerCampaignID = MAX([DialerCampaignID]),
    -- Measure: DistinctTargetGroupItemValueCount = COUNT(DISTINCT [TargetGroupItemValueID]),
    -- Measure: MinTargetGroupItemValueID = MIN([TargetGroupItemValueID]),
    -- Measure: MaxTargetGroupItemValueID = MAX([TargetGroupItemValueID]),
    -- Measure: DistinctCampaignCount = COUNT(DISTINCT [CampaignID]),
    -- Measure: MinCampaignID = MIN([CampaignID]),
    -- Measure: MaxCampaignID = MAX([CampaignID]),
    -- Measure: TotalNotFound = SUM([NotFound]),
    -- Measure: AverageNotFoundRate = AVG(CAST([NotFound] AS FLOAT)),
    -- Measure: MinNotFound = MIN([NotFound]),
    -- Measure: MaxNotFound = MAX([NotFound])
FROM dbo.DialerTask
GO

-- Fact: TargetGroupItemPeriod
-- Source: dbo.TargetGroupItemPeriod
CREATE OR ALTER VIEW semantic.TargetGroupItemPeriod AS
SELECT
    ID -- ID,
    TargetGroupItemValueID -- TargetGroupItemValueID,
    ContractProductID -- ContractProductID,
    Period -- Period,
    DMKey -- DMKey,
    DMTable -- DMTable,
    DMCharKey -- DMCharKey,
    -- Measure: TotalRecords = COUNT(*),
    -- Measure: DMKeyNonNullCount = COUNT(DMKey),
    -- Measure: DistinctTargetGroups = COUNT(DISTINCT DMKey),
    -- Measure: MinPeriod = MIN(Period),
    -- Measure: MaxPeriod = MAX(Period),
    -- Measure: AveragePeriod = AVG(Period),
    -- Measure: PeriodSpan = MAX(Period) - MIN(Period),
    -- Measure: MinDMKey = MIN(DMKey),
    -- Measure: MaxDMKey = MAX(DMKey),
    -- Measure: AverageDMKey = AVG(DMKey)
FROM dbo.TargetGroupItemPeriod
GO

-- Fact: Advertisement
-- Source: dbo.Advertisement
CREATE OR ALTER VIEW semantic.Advertisement AS
SELECT
    ID -- ID,
    ContractProductID -- ContractProductID,
    ProductID -- ProductID,
    VersionSectionID -- VersionSectionID,
    GeoAreaID -- GeoAreaID,
    HeadingID -- HeadingID,
    AdvBeginDate -- AdvBeginDate,
    AdvEndDate -- AdvEndDate,
    QA -- QA,
    CaseAttachmentID -- CaseAttachmentID,
    DMKey -- DMKey,
    DMTable -- DMTable,
    DMCharKey -- DMCharKey,
    Price -- Price,
    Discount -- Discount is a quantitative money field, not a boolean flag. Because it is NOT NULL, absence of a discount is represented by a zero value rather than NULL. To treat 'active' records as those with a discount applied, filter for Discount > 0. If your business rules differ (e.g., negative values mean something special), adjust the condition accordingly.,
    RulesDiscount -- RulesDiscount is a non-null monetary status indicator: it holds the discount amount applied by rules. Because the column is NOT NULL it will always contain a numeric value; 0 indicates no/ inactive discount, >0 indicates an active discount. Use RulesDiscount > 0 to select records with an active discount (or use RulesDiscount <> 0 if you want to treat unexpected negative values as active as well). If NULLs are found in practice, treat them as unknown/inactive and investigate ETL issues.,
    ProductKitID -- ProductKitID,
    IsSubscriptionBased -- IsSubscriptionBased is a non-nullable boolean flag indicating whether an advertisement is subscription-based. Because the column cannot be NULL, every row must have either 1 (yes) or 0 (no). To select subscription-based ads, filter with IsSubscriptionBased = 1. If NULLs appear in practice, investigate and treat them as data errors or unknowns.,
    IsVerified -- IsVerified is a boolean status flag indicating whether the advertisement has been verified. Because the column is defined NOT NULL, every row should contain either 1 (verified) or 0 (not verified). Use IsVerified = 1 to select verified/active advertisements. If you need to be defensive against unexpected NULLs, use ISNULL(IsVerified, 0) = 1 to treat NULL as false.,
    AlteredAmount -- AlteredAmount,
    DenyExtraction -- DenyExtraction,
    DesiredBeginDate -- DesiredBeginDate,
    DeactivationDate -- DeactivationDate,
    DeactivationReason -- DeactivationReason,
    ReactivatedContractID -- ReactivatedContractID,
    CompletedChanges -- CompletedChanges,
    PaymentDiscount -- PaymentDiscount is a nullable numeric flag that indicates whether a discount has been applied. NULL usually signals that no discount was recorded (unknown or not applicable). A populated decimal indicates the discount amount (or percent). To find records with an active discount, filter for non-NULL values greater than zero. If your business treats 0 as equivalent to NULL, adjust the filter accordingly (e.g., PaymentDiscount > 0). Also confirm whether the stored value is a currency or percent and whether negative values are possible or meaningful.,
    ProductDiscount -- ProductDiscount is a numeric indicator of whether an advertisement carries a discount. NULL indicates absence/unknownness of a discount; a populated value conveys the discount magnitude. To treat an advertisement as "active" in the sense of having a discount, filter for ProductDiscount IS NOT NULL AND ProductDiscount > 0. Verify business rules about zero or negative values and whether the column represents an absolute amount or a percentage; if any explicit presence (including zero) should be considered, use ProductDiscount IS NOT NULL, and if zero should be excluded use ProductDiscount <> 0.,
    BundleDiscount -- Although BundleDiscount is a numeric column (not a boolean flag), its semantics can be used as a status indicator: NULL = no discount specified / not applicable; a populated value = a discount is specified. To treat records with an active discount use a filter that requires a non-NULL, positive value. Verify business rules and units (absolute amount vs. percentage) and decide how to treat zero or negative values in your environment — e.g. use BundleDiscount > 0 to identify active discounts, or include = 0 if your domain considers an explicit zero differently.,
    BudgetAmount -- BudgetAmount,
    OneTimeAmount -- OneTimeAmount,
    ExtraFeeAmount -- ExtraFeeAmount,
    Months -- Months,
    SendTo -- SendTo,
    SendOn -- SendOn,
    CanExport -- CanExport,
    RelatedAdvertisementID -- RelatedAdvertisementID is a foreign-key-style status indicator: NULL means no relationship; a populated integer points at another Advertisement. Without documentation you cannot be certain whether populated values mark a record as inactive (e.g., superseded) or as part of an active relationship (e.g., a variant). Typical approaches: to get records that are likely primary/not superseded use RelatedAdvertisementID IS NULL; to get records that participate in relationships use RelatedAdvertisementID IS NOT NULL. Verify against referential constraints, triggers, or business rules (or look for companion columns like IsActive/StartDate/EndDate) before applying a production filter.,
    OneTimeAmountAfterDiscount -- Although this column is a monetary field rather than a traditional status flag, its NULL-vs-populated semantics act like a presence indicator: NULL = no applicable or unknown one‑time discounted charge; populated = there is a defined one‑time discounted charge (value shows the amount). To treat records as 'active' in the sense that they have a meaningful charge, filter for non-NULL and greater than zero. If your business defines 'active' differently (e.g., zero is considered active), adjust the SQL condition accordingly.,
    ExtraFeeAmountAfterDiscount -- This MONEY column holds the extra fee amount remaining after discounts. NULL generally indicates absence of a recorded extra fee (not applicable, not yet calculated, or missing). A non-NULL value is an explicit recorded amount; treat 0.00 as an explicit 'no extra fee' result. To select rows where an extra-fee value exists use: ExtraFeeAmountAfterDiscount IS NOT NULL. If your definition of 'active' excludes zero amounts, add: ExtraFeeAmountAfterDiscount IS NOT NULL AND ExtraFeeAmountAfterDiscount <> 0. Be cautious of negative values, currency context, and system behaviors that may use NULL vs 0 differently.,
    ActualSpentBudget -- ActualSpentBudget,
    ExtraFeeAmountAfterCC -- ExtraFeeAmountAfterCC,
    AlteredPPCAmountAfterCC -- AlteredPPCAmountAfterCC,
    IsCustomAttributesValid -- IsCustomAttributesValid is a nullable boolean status flag: a value of 1 marks the record as having valid custom attributes, 0 marks them as invalid, and NULL means the validity is unknown or hasn't been set. To select records considered 'active' with respect to this flag, filter for IsCustomAttributesValid = 1. If your application treats unknown (NULL) as valid or invalid, explicitly coalesce it (for example, WHERE ISNULL(IsCustomAttributesValid, 0) = 1 to treat NULL as false, or WHERE ISNULL(IsCustomAttributesValid, 1) = 1 to treat NULL as true).,
    DirProductId -- DirProductId,
    DirProductName -- DirProductName,
    -- Measure: TotalProductID = SUM(ProductID),
    -- Measure: TotalVersionSectionID = SUM(VersionSectionID),
    -- Measure: TotalGeoAreaID = SUM(GeoAreaID),
    -- Measure: TotalHeadingID = SUM(HeadingID),
    -- Measure: TotalDMKey = SUM(DMKey)
FROM dbo.Advertisement
WHERE
    Discount IS NULL AND RulesDiscount IS NULL AND IsVerified IS NULL AND PaymentDiscount IS NULL AND ProductDiscount IS NULL AND BundleDiscount IS NULL AND RelatedAdvertisementID IS NULL AND OneTimeAmountAfterDiscount IS NULL AND ExtraFeeAmountAfterDiscount IS NULL AND IsCustomAttributesValid IS NULL
    -- Filter for active records only
GO

-- Fact: CampaignNewLossCustomer
-- Source: dbo.CampaignNewLossCustomer
CREATE OR ALTER VIEW semantic.CampaignNewLossCustomer AS
SELECT
    ID -- ID,
    RuleID -- RuleID,
    BusinessPointID -- BusinessPointID,
    IsRequestedToSuspend -- IsRequestedToSuspend is a mandatory binary flag indicating whether the record has been requested to be suspended. Because the column is non‑nullable every row will contain either 0 or 1. Interpret 0 as the active/unsuspended state and 1 as suspended/requested-to-suspend. If there is any uncertainty about business semantics or if the flag meaning might be inverted, confirm with the data owner before relying on it in queries.,
    IsSuspended -- IsSuspended is a required boolean flag marking suspension state. Because it is NOT NULL every row must indicate suspended (1) or not suspended (0). To select records that are operational/active, filter for IsSuspended = 0. Any NULLs encountered are data anomalies and should be resolved according to your error-handling policy.,
    IsNewRecord -- IsNewRecord is a non-nullable boolean flag indicating whether the row is classified as a new record. Because the column cannot be NULL, every row must have either 0 or 1. Treat 1 as the positive/active state (new) and 0 as the negative/inactive state (not new); if you need the opposite definition for 'active' in your context, invert the condition. Verify business rules with domain owners if available.,
    ActionResult -- ActionResult,
    IsRequestedToResume -- IsRequestedToResume is a non-nullable boolean flag indicating whether a resume request exists. Because the column is NOT NULL, you should expect only 0 or 1 values; treat 1 as the 'active' or positive state (request present) and 0 as the negative state (no request). Any NULLs would signal data/schema issues and should be investigated.,
    DialerAction -- DialerAction,
    BatchID -- BatchID,
    -- Measure: TotalRecords = COUNT(*),
    -- Measure: DistinctRuleCount = COUNT(DISTINCT RuleID),
    -- Measure: MinRuleID = MIN(RuleID),
    -- Measure: MaxRuleID = MAX(RuleID),
    -- Measure: AvgRuleID = AVG(CAST(RuleID AS FLOAT)),
    -- Measure: DistinctBusinessPointCount = COUNT(DISTINCT BusinessPointID),
    -- Measure: MinBusinessPointID = MIN(BusinessPointID),
    -- Measure: MaxBusinessPointID = MAX(BusinessPointID),
    -- Measure: AvgBusinessPointID = AVG(CAST(BusinessPointID AS FLOAT)),
    -- Measure: DistinctDialerActionCount = COUNT(DISTINCT DialerAction),
    -- Measure: MinDialerAction = MIN(DialerAction),
    -- Measure: MaxDialerAction = MAX(DialerAction),
    -- Measure: AvgDialerAction = AVG(CAST(DialerAction AS FLOAT))
FROM dbo.CampaignNewLossCustomer
WHERE
    IsSuspended IS NULL AND IsNewRecord IS NULL AND IsRequestedToResume IS NULL
    -- Filter for active records only
GO

-- Fact: Classification
-- Source: dbo.Classification
CREATE OR ALTER VIEW semantic.Classification AS
SELECT
    ID -- ID,
    BusinessPointID -- BusinessPointID,
    ClassifierID -- ClassifierID,
    ClassifierNodeID -- ClassifierNodeID,
    DMKey -- DMKey,
    DMTable -- DMTable,
    DMCharKey -- DMCharKey,
    -- Measure: RecordCount = COUNT(*),
    -- Measure: UniqueDMKeyCount = COUNT(DISTINCT DMKey),
    -- Measure: TotalDMKey = SUM(DMKey),
    -- Measure: AverageDMKey = AVG(CAST(DMKey AS FLOAT)),
    -- Measure: MinDMKey = MIN(DMKey),
    -- Measure: MaxDMKey = MAX(DMKey)
FROM dbo.Classification
GO

-- Fact: ContractProduct
-- Source: dbo.ContractProduct
CREATE OR ALTER VIEW semantic.ContractProduct AS
SELECT
    ID -- ID,
    BusinessPointID -- BusinessPointID,
    RelatedContractID -- RelatedContractID,
    ProductID -- ProductID,
    State -- Because State is an INTEGER NOT NULL, every row contains a numeric state code; NULL is not allowed. That numeric value is a key into business semantics rather than a human-friendly label. To determine which numeric equals 'active' you should: (1) locate the canonical mapping (a lookup table such as dbo.ContractProductState or documentation); (2) if no mapping is available, inspect data (SELECT DISTINCT State FROM dbo.ContractProduct) and consult business owners. For filtering, use the integer code for 'active' (commonly 1 in many systems) or join/EXISTS against the lookup table that exposes an IsActive flag. Example safe patterns: (a) direct: WHERE State = <ACTIVE_CODE>; (b) lookup: WHERE EXISTS (SELECT 1 FROM dbo.ContractProductState s WHERE s.Code = dbo.ContractProduct.State AND s.IsActive = 1).,
    CreatedOn -- CreatedOn,
    CreatedBy -- CreatedBy,
    ModifiedOn -- ModifiedOn,
    ModifiedBy -- ModifiedBy,
    Price -- Price,
    IsSubscriptionBased -- IsSubscriptionBased is a non-nullable BIT flag indicating whether a ContractProduct is subscription-based. It contains only 0 or 1: 1 denotes subscription-based products, 0 denotes non-subscription products. Because the column is NOT NULL, you can safely filter for subscription products with IsSubscriptionBased = 1; if NULLs appear they should be investigated as data quality issues.,
    IsROI -- IsROI is a non‑nullable BIT status flag. Valid stored values are 1 (flagged/true) and 0 (not flagged/false). Because the column is NOT NULL, you normally do not need to guard against NULLs; if NULLs appear they represent data issues. To select records where the flag denotes the active/positive state, filter with IsROI = 1. Verify the precise business meaning of 'ROI' with domain owners if needed.,
    VersionSectionID -- VersionSectionID,
    SalesTypeID -- SalesTypeID,
    Quantity -- Quantity,
    Discount -- Discount is a non‑nullable monetary field, so every record contains a value. Interpret 0 as 'no discount' and >0 as 'discount active'. Because NULLs are not allowed by the schema, you do not need to handle missing values; if the business rules treat any non‑zero amount (including negatives) as active, use 'Discount <> 0' instead of 'Discount > 0'. If you need to detect originally unset discounts in a changed schema, a NULL would indicate unknown/not configured.,
    PaymentMethodID -- PaymentMethodID,
    RulesDiscount -- Although presented here in the status/flag analysis format, RulesDiscount is a numeric money field (NOT NULL) rather than a boolean flag. Because it cannot be NULL, presence/absence cannot be used to indicate state. Use the numeric value to infer state: treat values > 0 as an active discount, 0 as no discount, and handle negative values only after confirming business intent. If your business considers any non-zero value as active (including negatives), use RulesDiscount <> 0 instead of RulesDiscount > 0.,
    DMKey -- DMKey,
    DMTable -- DMTable,
    DMCharKey -- DMCharKey,
    SourceContractProductID -- SourceContractProductID,
    IsRenewed -- IsRenewed is a non-nullable boolean flag that records whether the contract product has been renewed. Because it cannot be NULL, every row must be marked either renewed (1) or not renewed (0). Typically an 'active' contract product is one that has not been superseded by a renewal, so filter with IsRenewed = 0 to get active records. If your business defines 'active' differently (for example, only renewed items are considered active), invert the condition to IsRenewed = 1.,
    IsRenewable -- IsRenewable is a non-nullable BIT flag indicating whether a ContractProduct can be renewed. Under normal circumstances you will never see NULL; 1 marks renewable products and 0 marks non-renewable ones. To select records that are 'active' in the sense of being renewable, filter with IsRenewable = 1. If you must defensively handle unexpected NULLs, include an explicit NULL check (e.g. IsRenewable = 1 OR IsRenewable IS NULL) depending on how you want to treat unknowns.,
    InterruptionDate -- InterruptionDate,
    AlteredAmount -- AlteredAmount,
    CancelledOn -- CancelledOn is a soft‑status timestamp: NULL denotes no cancellation (active), a non‑NULL value denotes cancellation at that datetime. When filtering for active records prefer treating both NULL and future cancellations as active (e.g. WHERE CancelledOn IS NULL OR CancelledOn > GETUTCDATE()). Be aware of timezone considerations (use GETDATE() vs GETUTCDATE() consistently with your data), possible default sentinel values, and that other date fields (effective/expiry) or status columns may also affect what you consider fully active.,
    CancelledBy -- CancelledBy is a nullable NVARCHAR flag that appears to record who cancelled the ContractProduct. NULL typically means no cancellation was recorded (the record is active). A non-NULL/non-empty value is the user or system account that cancelled the row and therefore indicates the record is cancelled (soft-deleted). When filtering for active records, check for NULL or empty/whitespace values as shown. Note: if the schema also has a cancellation timestamp or a separate boolean, use those for stronger guarantees; some systems may use special values (e.g., 'AdminUpdate') or might populate this column for other administrative actions, so validate against related audit/date columns if available.,
    PaymentDiscount -- Treat NULL as 'no discount defined / not applicable'. A non-NULL numeric value means an explicit discount has been recorded. If you consider only positive discounts as active, use PaymentDiscount IS NOT NULL AND PaymentDiscount > 0. The provided active_filter (PaymentDiscount IS NOT NULL AND PaymentDiscount <> 0) finds rows with any explicit non-zero discount; adjust to >0 if negative values should be excluded.,
    ProductDiscount -- ProductDiscount is a numeric field (DECIMAL(18,3)), not a boolean flag. NULL indicates no discount has been recorded (unspecified/unknown). A non-NULL value records the discount amount or rate. To find records with an active discount, filter for non-NULL and greater than zero (ProductDiscount IS NOT NULL AND ProductDiscount > 0). If your definition of "active" is any explicitly set discount (including zero or negative), use ProductDiscount IS NOT NULL or ProductDiscount <> 0 to capture any nonzero explicit values. If NULL should be treated as zero, use COALESCE(ProductDiscount, 0) > 0.,
    BundleDiscount -- Treat NULL as 'no discount specified / not applicable.' A populated decimal means a discount has been set; interpret the numeric magnitude per business rules (amount vs percent). To find rows with an effective/active bundle discount, filter for non‑NULL and non‑zero values. If your business considers zero as active or uses negative values, adjust the condition accordingly.,
    BudgetAmount -- BudgetAmount,
    OneTimeAmount -- OneTimeAmount,
    ExtraFeeAmount -- ExtraFeeAmount,
    Months -- Months,
    OneTimeAmountAfterDiscount -- OneTimeAmountAfterDiscount is a nullable monetary field. NULL typically means no amount was set (not applicable or not provided). A populated value is the actual post‑discount one‑time charge. Because it conveys an amount rather than a state, you cannot reliably use it alone to determine whether a record is 'active' unless your business rule defines activity by presence/size of this amount. If such a rule exists, filter for IS NOT NULL and/or <> 0 as appropriate; otherwise use explicit status/date fields to determine activity.,
    ExtraFeeAmountAfterDiscount -- This column is a monetary measure, not a binary status flag. NULL indicates absence/not-applicable/uncomputed. A non-NULL value gives the actual extra-fee amount after discounts; distinguish explicit zeros from NULLs in queries. If you need to treat rows with an applied extra fee as 'active' for this indicator, filter for non-NULL and non-zero values as shown. If your business considers explicit zero as active (or wants to include computed zeros), adjust the filter accordingly (e.g., ExtraFeeAmountAfterDiscount IS NOT NULL). Also validate unexpected negative values with data quality rules.,
    ActualSpentBudget -- ActualSpentBudget,
    ExtraFeeAmountAfterCC -- ExtraFeeAmountAfterCC,
    AlteredPPCAmountAfterCC -- AlteredPPCAmountAfterCC,
    IsInterruptedFromCRM -- IsInterruptedFromCRM is a non-nullable boolean flag where 1 signals the product was interrupted in CRM and 0 signals it was not. Because the column is NOT NULL, expect only 0/1 values; NULL would be an anomaly (or represent 'unknown' only if the schema allowed it). To select active (not interrupted) records filter for IsInterruptedFromCRM = 0. If you must defensively handle unexpected NULLs, use (IsInterruptedFromCRM = 0 OR IsInterruptedFromCRM IS NULL) to treat NULL as active.,
    -- Measure: TotalState = SUM(State),
    -- Measure: TotalPrice = SUM(Price),
    -- Measure: TotalSalesTypeID = SUM(SalesTypeID),
    -- Measure: TotalQuantity = SUM(Quantity),
    -- Measure: TotalDiscount = SUM(Discount)
FROM dbo.ContractProduct
WHERE
    State IS NULL AND IsROI IS NULL AND Discount IS NULL AND RulesDiscount IS NULL AND IsRenewed IS NULL AND IsRenewable IS NULL AND CancelledOn IS NULL AND CancelledBy IS NULL AND PaymentDiscount IS NULL AND ProductDiscount IS NULL AND BundleDiscount IS NULL AND OneTimeAmountAfterDiscount IS NULL AND ExtraFeeAmountAfterDiscount IS NULL AND IsInterruptedFromCRM IS NULL
    -- Filter for active records only
GO

-- Fact: CaseComment
-- Source: dbo.CaseComment
CREATE OR ALTER VIEW semantic.CaseComment AS
SELECT
    ID -- ID,
    CaseID -- CaseID,
    CommentText -- CommentText,
    CreatedOn -- CreatedOn,
    CreatedBy -- CreatedBy,
    TaskID -- TaskID,
    CommentType -- CommentType,
    DMKey -- DMKey,
    DMTable -- DMTable,
    DMCharKey -- DMCharKey,
    -- Measure: TotalComments = COUNT(*),
    -- Measure: CommentType_Count = COUNT(CommentType),
    -- Measure: CommentType_DistinctCount = COUNT(DISTINCT CommentType),
    -- Measure: CommentType_Min = MIN(CommentType),
    -- Measure: CommentType_Max = MAX(CommentType),
    -- Measure: CommentType_Average = AVG(CAST(CommentType AS FLOAT)),
    -- Measure: DMKey_Count = COUNT(DMKey),
    -- Measure: DMKey_DistinctCount = COUNT(DISTINCT DMKey),
    -- Measure: DMKey_Min = MIN(DMKey),
    -- Measure: DMKey_Max = MAX(DMKey)
FROM dbo.CaseComment
GO

-- Fact: CampaignScoreHistoryDetail
-- Source: dbo.CampaignScoreHistoryDetail
CREATE OR ALTER VIEW semantic.CampaignScoreHistoryDetail AS
SELECT
    ID -- ID,
    CampaignScoreHistoryID -- CampaignScoreHistoryID is a required identifier that ties this detail record to a CampaignScoreHistory parent. Because it is NOT NULL, you will not see NULLs in the column; a populated integer indicates the parent linkage. If you want to treat 'active' as rows that have a valid parent, filter by CampaignScoreHistoryID IS NOT NULL (or CampaignScoreHistoryID > 0). For stronger guarantees, join to dbo.CampaignScoreHistory and apply any status flags on the parent (for example, only include rows where the parent exists and is not marked deleted).,
    CampaignBucketID -- CampaignBucketID,
    ScoreMainActivity -- ScoreMainActivity,
    ScoreSeniority -- ScoreSeniority,
    ScoreValueBillingRange -- ScoreValueBillingRange,
    ScoreCombinationMedium -- ScoreCombinationMedium,
    ScoreMaxInvestmentMedium -- ScoreMaxInvestmentMedium,
    ScoreCombinationProduct -- ScoreCombinationProduct,
    ScoreCustomerBehavior -- ScoreCustomerBehavior,
    ScoreSalesmanManagement -- ScoreSalesmanManagement,
    ScoreTotal -- ScoreTotal,
    TargetingAmountCalculated -- TargetingAmountCalculated,
    TargetingAmountByUser -- TargetingAmountByUser,
    Characterization -- Characterization,
    ScoreDifferenceToAvg -- ScoreDifferenceToAvg,
    CampaignID -- CampaignID,
    BusinessPointID -- BusinessPointID,
    BussinessPointCode -- BussinessPointCode,
    BusinessPointName -- BusinessPointName,
    ProductMixture -- ProductMixture,
    VerticalsGrouping -- VerticalsGrouping,
    PrintAmount -- PrintAmount,
    PPCAmount -- PPCAmount,
    XOGRAmount -- XOGRAmount,
    WebSiteAmount -- WebSiteAmount,
    TotalAmount -- TotalAmount,
    NewChannel -- NewChannel,
    NewChannelID -- NewChannelID,
    NewSalesmanID -- NewSalesmanID,
    NewSalesmanCode -- NewSalesmanCode,
    NewSalesmanName -- NewSalesmanName,
    FirstCustomerExpirationMonth -- FirstCustomerExpirationMonth,
    SegmentID -- SegmentID,
    SegmentDescription -- SegmentDescription,
    FileID -- FileID,
    ItemStatus -- ItemStatus is a nullable integer used to carry a coded status. NULL means no explicit status was assigned (unknown, default, or not processed). A populated integer maps to a specific state; you must consult the status-code documentation or lookup table to interpret each value. For querying 'active' rows, prefer using the explicit active code (ItemStatus = <active_code>). If the code mapping is unavailable and the business rule treats any assigned status as active, use ItemStatus IS NOT NULL. Consider adding or documenting a status lookup table and avoiding NULLs for clearer semantics.,
    WorkingStatus -- WorkingStatus is a nullable integer flag that encodes the record's state. NULL indicates the state is absent/unknown/not set; any non-NULL integer corresponds to an enumerated status whose semantics must be obtained from the application or a reference table. To select active records, filter for the integer value(s) that represent 'active' (commonly 1), and explicitly decide whether NULL should be included or excluded based on business rules.,
    TargetGroupItemValueID -- TargetGroupItemValueID,
    MediumCombination -- MediumCombination,
    MaxInvestmentMedium -- MaxInvestmentMedium,
    ScoreSalesmanManagementType -- ScoreSalesmanManagementType,
    ScoreCustomerBehaviorType -- ScoreCustomerBehaviorType,
    SupervisorID -- SupervisorID is a nullable indicator of supervisor assignment. NULL usually means 'no supervisor assigned / unknown / removed'. A non-NULL integer is the assigned supervisor's ID. To consider a record 'active' in the sense of having an assigned supervisor, filter with WHERE SupervisorID IS NOT NULL. If your environment uses a sentinel (such as 0) for unassigned, tighten the filter to WHERE SupervisorID IS NOT NULL AND SupervisorID <> 0.,
    SupervisorName -- SupervisorName is an ownership/assignment field, not a boolean flag. NULL indicates absence of an assigned/known supervisor; a populated string indicates the supervisor recorded for that history detail. To treat a row as having an active/assigned supervisor, exclude NULLs, empty or whitespace-only strings, and known sentinel values such as 'Not available' (see active_filter). Because the column uses a case-insensitive collation (Greek_CI_AS), string comparisons are case-insensitive; if you have additional sentinel values (e.g. 'TBD', 'Unknown'), add them to the filter.,
    PreviousSalesmanID -- PreviousSalesmanID,
    PreviousSalesmanCode -- PreviousSalesmanCode,
    PreviousSalesmanName -- PreviousSalesmanName,
    MainActivityID -- MainActivityID,
    MainActivity -- MainActivity,
    BucketGroup -- BucketGroup,
    -- Measure: RecordCount = COUNT(ID),
    -- Measure: DistinctCampaignCount = COUNT(DISTINCT CampaignID),
    -- Measure: DistinctBusinessPointCount = COUNT(DISTINCT BusinessPointID),
    -- Measure: DistinctFileCount = COUNT(DISTINCT FileID),
    -- Measure: DistinctSegmentCount = COUNT(DISTINCT SegmentID),
    -- Measure: TotalTargetingAmountCalculated = SUM(TargetingAmountCalculated),
    -- Measure: AverageTargetingAmountCalculated = AVG(TargetingAmountCalculated),
    -- Measure: MinTargetingAmountCalculated = MIN(TargetingAmountCalculated),
    -- Measure: MaxTargetingAmountCalculated = MAX(TargetingAmountCalculated),
    -- Measure: TotalTargetingAmountByUser = SUM(TargetingAmountByUser),
    -- Measure: AverageTargetingAmountByUser = AVG(TargetingAmountByUser),
    -- Measure: MinTargetingAmountByUser = MIN(TargetingAmountByUser),
    -- Measure: MaxTargetingAmountByUser = MAX(TargetingAmountByUser),
    -- Measure: TotalPrintAmount = SUM(PrintAmount),
    -- Measure: AveragePrintAmount = AVG(PrintAmount),
    -- Measure: MinPrintAmount = MIN(PrintAmount),
    -- Measure: MaxPrintAmount = MAX(PrintAmount),
    -- Measure: TotalPPCAmount = SUM(PPCAmount),
    -- Measure: AveragePPCAmount = AVG(PPCAmount),
    -- Measure: TotalXOGRAmount = SUM(XOGRAmount),
    -- Measure: AverageXOGRAmount = AVG(XOGRAmount),
    -- Measure: TotalWebSiteAmount = SUM(WebSiteAmount),
    -- Measure: AverageWebSiteAmount = AVG(WebSiteAmount),
    -- Measure: TotalAmount = SUM(TotalAmount),
    -- Measure: AverageTotalAmount = AVG(TotalAmount),
    -- Measure: MinTotalAmount = MIN(TotalAmount),
    -- Measure: MaxTotalAmount = MAX(TotalAmount),
    -- Measure: TotalScoreTotal = SUM(ScoreTotal),
    -- Measure: AverageScoreTotal = AVG(ScoreTotal),
    -- Measure: MinScoreTotal = MIN(ScoreTotal),
    -- Measure: MaxScoreTotal = MAX(ScoreTotal),
    -- Measure: AverageScoreMainActivity = AVG(ScoreMainActivity),
    -- Measure: MinScoreMainActivity = MIN(ScoreMainActivity),
    -- Measure: MaxScoreMainActivity = MAX(ScoreMainActivity),
    -- Measure: AverageScoreSeniority = AVG(ScoreSeniority),
    -- Measure: MinScoreSeniority = MIN(ScoreSeniority),
    -- Measure: MaxScoreSeniority = MAX(ScoreSeniority),
    -- Measure: AverageScoreValueBillingRange = AVG(ScoreValueBillingRange),
    -- Measure: MinScoreValueBillingRange = MIN(ScoreValueBillingRange),
    -- Measure: MaxScoreValueBillingRange = MAX(ScoreValueBillingRange),
    -- Measure: AverageScoreCombinationMedium = AVG(ScoreCombinationMedium),
    -- Measure: MinScoreCombinationMedium = MIN(ScoreCombinationMedium),
    -- Measure: MaxScoreCombinationMedium = MAX(ScoreCombinationMedium),
    -- Measure: AverageScoreMaxInvestmentMedium = AVG(ScoreMaxInvestmentMedium),
    -- Measure: MinScoreMaxInvestmentMedium = MIN(ScoreMaxInvestmentMedium),
    -- Measure: MaxScoreMaxInvestmentMedium = MAX(ScoreMaxInvestmentMedium),
    -- Measure: AverageScoreCombinationProduct = AVG(ScoreCombinationProduct),
    -- Measure: MinScoreCombinationProduct = MIN(ScoreCombinationProduct),
    -- Measure: MaxScoreCombinationProduct = MAX(ScoreCombinationProduct),
    -- Measure: AverageScoreCustomerBehavior = AVG(ScoreCustomerBehavior),
    -- Measure: MinScoreCustomerBehavior = MIN(ScoreCustomerBehavior),
    -- Measure: MaxScoreCustomerBehavior = MAX(ScoreCustomerBehavior),
    -- Measure: AverageScoreSalesmanManagement = AVG(ScoreSalesmanManagement),
    -- Measure: MinScoreSalesmanManagement = MIN(ScoreSalesmanManagement),
    -- Measure: MaxScoreSalesmanManagement = MAX(ScoreSalesmanManagement),
    -- Measure: AverageScoreDifferenceToAvg = AVG(ScoreDifferenceToAvg),
    -- Measure: MinScoreDifferenceToAvg = MIN(ScoreDifferenceToAvg),
    -- Measure: MaxScoreDifferenceToAvg = MAX(ScoreDifferenceToAvg),
    -- Measure: SumTargetingCalculatedMinusByUser = SUM(TargetingAmountCalculated - TargetingAmountByUser),
    -- Measure: CountByItemStatus = COUNT(ItemStatus),
    -- Measure: CountByWorkingStatus = COUNT(WorkingStatus),
    -- Measure: MinCharacterization = MIN(Characterization),
    -- Measure: MaxCharacterization = MAX(Characterization)
FROM dbo.CampaignScoreHistoryDetail
WHERE
    CampaignScoreHistoryID IS NULL AND ItemStatus IS NULL AND WorkingStatus IS NULL AND SupervisorID IS NULL AND SupervisorName IS NULL
    -- Filter for active records only
GO

-- Fact: PointRelationship
-- Source: dbo.PointRelationship
CREATE OR ALTER VIEW semantic.PointRelationship AS
SELECT
    BusinessPointID -- BusinessPointID,
    TargetBussinessPointID -- TargetBussinessPointID,
    Type -- Type,
    DMKey -- DMKey,
    DMTable -- DMTable,
    DMCharKey -- DMCharKey,
    -- Measure: RecordCount = COUNT(*),
    -- Measure: DistinctDMKeyCount = COUNT(DISTINCT DMKey),
    -- Measure: DistinctTypeCount = COUNT(DISTINCT Type),
    -- Measure: MinType = MIN(Type),
    -- Measure: MaxType = MAX(Type),
    -- Measure: AvgType = AVG(CAST(Type AS FLOAT)),
    -- Measure: MinDMKey = MIN(DMKey),
    -- Measure: MaxDMKey = MAX(DMKey),
    -- Measure: AvgDMKey = AVG(CAST(DMKey AS FLOAT))
FROM dbo.PointRelationship
GO

-- Fact: BusinessPointConfirmationHistory
-- Source: dbo.BusinessPointConfirmationHistory
CREATE OR ALTER VIEW semantic.BusinessPointConfirmationHistory AS
SELECT
    ID -- ID,
    BusinessPointID -- BusinessPointID,
    BusinessPointIdentificationID -- BusinessPointIdentificationID,
    TimeStamp -- TimeStamp,
    PerformedByID -- PerformedByID,
    ActionType -- ActionType,
    TaskID -- TaskID,
    SourceType -- SourceType,
    IsThirdPartyDataSource -- IsThirdPartyDataSource is a non-nullable boolean flag indicating whether a BusinessPointConfirmationHistory row originated from a third‑party data source. Valid values are 1 (third‑party) and 0 (not third‑party). NULL is not expected; if present it would signal an unexpected/missing origin. To select records marked as true/"active" for this flag, filter with IsThirdPartyDataSource = 1 (or use = 0 to get non‑third‑party rows).,
    -- Measure: CountConfirmations = COUNT(*),
    -- Measure: DistinctBusinessPointCount = COUNT(DISTINCT BusinessPointIdentificationID),
    -- Measure: DistinctPerformerCount = COUNT(DISTINCT PerformedByID),
    -- Measure: DistinctTaskCount = COUNT(DISTINCT TaskID),
    -- Measure: DistinctActionTypeCount = COUNT(DISTINCT ActionType),
    -- Measure: DistinctSourceTypeCount = COUNT(DISTINCT SourceType),
    -- Measure: MinBusinessPointIdentificationID = MIN(BusinessPointIdentificationID),
    -- Measure: MaxBusinessPointIdentificationID = MAX(BusinessPointIdentificationID),
    -- Measure: MinPerformedByID = MIN(PerformedByID),
    -- Measure: MaxPerformedByID = MAX(PerformedByID),
    -- Measure: MinTaskID = MIN(TaskID),
    -- Measure: MaxTaskID = MAX(TaskID),
    -- Measure: MinActionType = MIN(ActionType),
    -- Measure: MaxActionType = MAX(ActionType),
    -- Measure: AverageActionType = AVG(ActionType),
    -- Measure: MinSourceType = MIN(SourceType),
    -- Measure: MaxSourceType = MAX(SourceType)
FROM dbo.BusinessPointConfirmationHistory
WHERE
    IsThirdPartyDataSource IS NULL
    -- Filter for active records only
GO

-- Fact: UserLog
-- Source: dbo.UserLog
CREATE OR ALTER VIEW semantic.UserLog AS
SELECT
    ID -- ID,
    UserID -- UserID,
    Timestamp -- Timestamp,
    PC -- PC,
    IP -- IP,
    Action -- Action,
    User_Version -- User_Version,
    -- Measure: RecordCount = COUNT(*),
    -- Measure: ActionNonNullCount = COUNT(Action),
    -- Measure: DistinctActionCount = COUNT(DISTINCT Action),
    -- Measure: SumAction = SUM(Action),
    -- Measure: AverageAction = AVG(Action),
    -- Measure: MinAction = MIN(Action),
    -- Measure: MaxAction = MAX(Action)
FROM dbo.UserLog
GO

-- Fact: Ticket
-- Source: dbo.Ticket
CREATE OR ALTER VIEW semantic.Ticket AS
SELECT
    ID -- ID,
    BusinessPointID -- BusinessPointID,
    Result -- Result,
    TicketTypeID -- TicketTypeID,
    NeedEval -- NeedEval,
    PriorityID -- PriorityID,
    SourceTypeID -- SourceTypeID,
    ResultComm -- ResultComm,
    ResultCommDate -- ResultCommDate,
    Status -- Status is a required integer code representing the ticket's lifecycle state. Because the column is NOT NULL, you will always see a numeric code; NULL is not used. To identify active tickets you must filter by the specific code(s) that denote non-terminal/working states in your environment. If you do not know the codes, either consult the status lookup/metadata or use the inverse approach to exclude terminal statuses: WHERE Status NOT IN (<closed_cancelled_codes>). Always use explicit code lists rather than relying on presence/absence of NULL.,
    CreatedOn -- CreatedOn,
    CreatedByID -- CreatedByID,
    NeedSupervisor -- NeedSupervisor is a non-nullable boolean flag indicating whether a ticket requires supervisor intervention. Because the column is NOT NULL, every row will contain either 0 or 1. Use NeedSupervisor = 1 to select tickets that are flagged (require supervisor), and NeedSupervisor = 0 to select tickets that are not flagged.,
    ResultCustomer -- ResultCustomer,
    CompletedOn -- CompletedOn,
    CompletedByID -- CompletedByID,
    ResultCommTypeID -- ResultCommTypeID,
    CancellationResonID -- CancellationResonID,
    ChargeOfResponsibilityID -- ChargeOfResponsibilityID,
    RequestReasonID -- RequestReasonID,
    ResolutionMethodID -- ResolutionMethodID,
    RelatedContractID -- RelatedContractID,
    RelatedSalesmanID -- RelatedSalesmanID,
    HandlerID -- HandlerID,
    NeedCancelContractEval -- NeedCancelContractEval,
    DMKey -- DMKey,
    DMTable -- DMTable,
    DMCharKey -- DMCharKey,
    LastAssignUserID -- LastAssignUserID,
    LastAssignRoleID -- LastAssignRoleID,
    ContactName -- ContactName,
    ReceivedOn -- ReceivedOn,
    CreditNumber -- CreditNumber,
    CreditAmount -- CreditAmount,
    CancelAmount -- CancelAmount,
    DPSCurrentNodeID -- DPSCurrentNodeID,
    TargetCampaignID -- TargetCampaignID,
    TargetSubChannelID -- TargetSubChannelID,
    TargetSalesmanID -- TargetSalesmanID,
    TargetSupervisorID -- TargetSupervisorID is a nullable foreign-key style status column. NULL indicates there is no specific supervisor target (unassigned or not applicable). A populated integer value indicates the ticket is targeted/assigned to the supervisor with that ID. To treat 'active' records as tickets that have a target supervisor, filter for TargetSupervisorID IS NOT NULL. If you need to ensure the referenced supervisor actually exists and is not soft-deleted, join to the supervisor table and add that table's active flag (for example: JOIN dbo.Supervisor s ON Ticket.TargetSupervisorID = s.SupervisorID WHERE Ticket.TargetSupervisorID IS NOT NULL AND s.IsActive = 1). Also verify whether your system ever uses sentinel values (0 or negative) to mean unassigned and adjust the filter accordingly.,
    MaterialDeliveryDate -- MaterialDeliveryDate,
    IsEditContractType -- IsEditContractType is a non-nullable boolean status flag. Because the column is NOT NULL, you will never see NULL; every row will have either 0 or 1. Interpret 1 as the flag being active/true and 0 as inactive/false. To select rows where this flag is active, filter with IsEditContractType = 1. If the precise business meaning (allowed vs. performed) is important, validate with application or product owners.,
    InParallelRoleID -- InParallelRoleID,
    RelatedAdvertisementID -- RelatedAdvertisementID is a nullable foreign-key-style status indicator: NULL signals no associated advertisement, a populated integer is the ID of the related advertisement. To treat "active" as "has an associated advertisement", filter with RelatedAdvertisementID IS NOT NULL; to be strict, also validate the reference exists via a JOIN or EXISTS against the Advertisement table.,
    InitialTicketTypeID -- InitialTicketTypeID,
    SerialTicketTypdeID -- SerialTicketTypdeID,
    SerialRoleID -- SerialRoleID,
    RequestedChanges -- RequestedChanges,
    WebChangesContractID -- WebChangesContractID,
    SelectedSalesmanID -- SelectedSalesmanID,
    Code -- Code,
    CampaignLeadId -- CampaignLeadId,
    BusinessPointIdentificationId -- BusinessPointIdentificationId,
    -- Measure: TicketCount = COUNT(DMKey),
    -- Measure: DistinctResultCount = COUNT(DISTINCT Result),
    -- Measure: ResultMin = MIN(Result),
    -- Measure: ResultMax = MAX(Result),
    -- Measure: DistinctStatusCount = COUNT(DISTINCT Status),
    -- Measure: StatusMin = MIN(Status),
    -- Measure: StatusMax = MAX(Status),
    -- Measure: DMKeyMin = MIN(DMKey),
    -- Measure: DMKeyMax = MAX(DMKey),
    -- Measure: DistinctLastAssignUserCount = COUNT(DISTINCT LastAssignUserID),
    -- Measure: LastAssignUserIDMin = MIN(LastAssignUserID),
    -- Measure: DistinctLastAssignRoleCount = COUNT(DISTINCT LastAssignRoleID),
    -- Measure: LastAssignRoleIDMin = MIN(LastAssignRoleID),
    -- Measure: TotalCreditNumber = SUM(CreditNumber),
    -- Measure: AverageCreditNumber = AVG(CreditNumber),
    -- Measure: TotalCreditAmount = SUM(CreditAmount),
    -- Measure: AverageCreditAmount = AVG(CreditAmount),
    -- Measure: MinCreditAmount = MIN(CreditAmount),
    -- Measure: MaxCreditAmount = MAX(CreditAmount),
    -- Measure: TotalCancelAmount = SUM(CancelAmount),
    -- Measure: AverageCancelAmount = AVG(CancelAmount),
    -- Measure: MinCancelAmount = MIN(CancelAmount),
    -- Measure: MaxCancelAmount = MAX(CancelAmount),
    -- Measure: DistinctDPSCurrentNodeCount = COUNT(DISTINCT DPSCurrentNodeID),
    -- Measure: DPSCurrentNodeIDMin = MIN(DPSCurrentNodeID),
    -- Measure: DistinctRelatedAdvertisementCount = COUNT(DISTINCT RelatedAdvertisementID),
    -- Measure: RelatedAdvertisementIDMin = MIN(RelatedAdvertisementID),
    -- Measure: DistinctInitialTicketTypeCount = COUNT(DISTINCT InitialTicketTypeID),
    -- Measure: InitialTicketTypeIDMin = MIN(InitialTicketTypeID),
    -- Measure: DistinctSerialTicketTypeCount = COUNT(DISTINCT SerialTicketTypdeID),
    -- Measure: SerialTicketTypdeIDMin = MIN(SerialTicketTypdeID),
    -- Measure: DistinctSerialRoleCount = COUNT(DISTINCT SerialRoleID),
    -- Measure: SerialRoleIDMin = MIN(SerialRoleID),
    -- Measure: TotalRequestedChanges = SUM(RequestedChanges),
    -- Measure: AverageRequestedChanges = AVG(RequestedChanges),
    -- Measure: MinRequestedChanges = MIN(RequestedChanges),
    -- Measure: MaxRequestedChanges = MAX(RequestedChanges),
    -- Measure: DistinctSelectedSalesmanCount = COUNT(DISTINCT SelectedSalesmanID),
    -- Measure: SelectedSalesmanIDMin = MIN(SelectedSalesmanID)
FROM dbo.Ticket
WHERE
    Status IS NULL AND TargetSupervisorID IS NULL AND IsEditContractType IS NULL AND RelatedAdvertisementID IS NULL
    -- Filter for active records only
GO

-- Fact: TicketResult
-- Source: dbo.TicketResult
CREATE OR ALTER VIEW semantic.TicketResult AS
SELECT
    ID -- ID,
    TicketID -- TicketID,
    TaskID -- TaskID,
    Result -- Result,
    ResultComm -- ResultComm,
    ResultCommDate -- ResultCommDate,
    ResultCustomer -- ResultCustomer,
    PendingReview -- PendingReview,
    ResultCommTypeID -- ResultCommTypeID,
    ResolutionMethodID -- ResolutionMethodID,
    DMKey -- DMKey,
    DMTable -- DMTable,
    DMCharKey -- DMCharKey,
    PerformedOn -- PerformedOn,
    PerformedByID -- PerformedByID,
    InParallelRoleID -- InParallelRoleID,
    -- Measure: TotalResult = SUM(Result),
    -- Measure: AverageResult = AVG(Result),
    -- Measure: MinResult = MIN(Result),
    -- Measure: MaxResult = MAX(Result),
    -- Measure: CountResults = COUNT(Result),
    -- Measure: DistinctResultCount = COUNT(DISTINCT Result),
    -- Measure: CountDMKey = COUNT(DMKey),
    -- Measure: DistinctDMKeyCount = COUNT(DISTINCT DMKey),
    -- Measure: MinDMKey = MIN(DMKey),
    -- Measure: MaxDMKey = MAX(DMKey),
    -- Measure: CountPerformedBy = COUNT(PerformedByID),
    -- Measure: DistinctPerformers = COUNT(DISTINCT PerformedByID),
    -- Measure: MinPerformedByID = MIN(PerformedByID),
    -- Measure: MaxPerformedByID = MAX(PerformedByID),
    -- Measure: RecordCount = COUNT(*)
FROM dbo.TicketResult
GO

-- Fact: FreeTemplate
-- Source: dbo.FreeTemplate
CREATE OR ALTER VIEW semantic.FreeTemplate AS
SELECT
    ID -- ID,
    BusinessPointID -- BusinessPointID,
    ProductID -- ProductID,
    State -- State is a non-nullable integer status code. Because NULL is not permitted, you should not rely on NULL to mean unknown or not applicable; encountering NULL would signal a data quality issue. Populated integers represent one of the predefined status codes; the exact mapping must be obtained from the system documentation or a reference table. To select active records, use an equality filter for the integer code that represents 'active' (commonly State = 1); if your system uses multiple active codes use State IN (...), or if 0 means inactive use State <> 0. Confirm the correct code(s) before applying the filter.,
    CreatedOn -- CreatedOn,
    CreatedBy -- CreatedBy,
    ModifiedOn -- ModifiedOn,
    ModifiedBy -- ModifiedBy,
    Price -- Price,
    IsSubscriptionBased -- IsSubscriptionBased is a required boolean flag indicating whether a FreeTemplate is subscription-based. Because the column is NOT NULL every row will have either 0 or 1. Interpret 1 as the positive/"on" state (subscription-based) and 0 as the negative/"off" state (not subscription-based). To select subscription-based records filter using IsSubscriptionBased = 1; to select non-subscription records use IsSubscriptionBased = 0.,
    IsROI -- IsROI is a non‑nullable BIT status flag. Because it cannot be NULL, every row will contain either 0 or 1. Interpret 1 as the positive/active state (the template is an ROI) and 0 as the negative/inactive state (the template is not an ROI). To return 'active' (ROI) records filter with WHERE IsROI = 1.,
    VersionSectionID -- VersionSectionID,
    SalesTypeID -- SalesTypeID,
    Quantity -- Quantity,
    Discount -- Discount is a non‑nullable monetary field that represents the discount amount for the template. Because the column cannot be NULL, you will always see a numeric value; interpret 0 as no discount and >0 as an active discount. If your business treats negative amounts differently (e.g., as reductions stored as negative numbers), adjust the active filter accordingly (for example Discount <> 0 or Discount < 0).,
    PaymentMethodID -- PaymentMethodID,
    RulesDiscount -- RulesDiscount is a non-null money value that functions as a numeric status: non‑zero (typically > 0) means a discount is applied, and 0 means no discount. Because the column cannot be NULL, do not rely on NULL checks. If your business rules allow negative discounts or treat any non-zero as active, use 'RulesDiscount <> 0' instead of 'RulesDiscount > 0'.,
    DMKey -- DMKey,
    DMTable -- DMTable,
    DMCharKey -- DMCharKey,
    SourceContractProductID -- SourceContractProductID,
    IsRenewed -- IsRenewed is a boolean status flag (BIT NOT NULL) indicating whether a FreeTemplate record has been renewed. Because the column is NOT NULL every row will have either 0 or 1. Use IsRenewed = 1 to select records that are 'renewed/active' (or IsRenewed = 0 to select 'not renewed' records). If your business definition of 'active' differs, adjust the filter accordingly.,
    IsRenewable -- IsRenewable is a Boolean status flag (BIT NOT NULL). Because it cannot be NULL, every row will explicitly indicate renewable (1) or not renewable (0). To select records considered 'active' with respect to renewable status, filter for IsRenewable = 1. If you need to treat the value as a proper boolean in client code, interpret 1 as true and 0 as false.,
    InterruptionDate -- InterruptionDate,
    AlteredAmount -- AlteredAmount,
    CancelledOn -- CancelledOn is a cancellation timestamp. NULL indicates no cancellation has been recorded (the item is considered active unless other rules apply). A non-NULL datetime is the moment the item was/will be cancelled; past datetimes mean already cancelled, future datetimes mean a scheduled cancellation. To treat scheduled future cancellations as still active you would use CancelledOn IS NULL OR CancelledOn > GETUTCDATE(), otherwise use CancelledOn IS NULL to select currently active (not cancelled) records.,
    CancelledBy -- CancelledBy is a cancellation flag stored as the cancelling user's name/ID. If NULL (or an empty string) there is no recorded cancellation and the template is considered active. If the column contains a non-empty value, it represents who performed the cancellation and the record should be treated as cancelled. If your environment uses special placeholders (e.g. 'N/A', 'Unknown'), extend the filter to exclude those values as needed.,
    PaymentDiscount -- PaymentDiscount is a numeric status-like column indicating whether a discount has been recorded for the record. NULL = no discount configured / not applicable / unknown. A populated value = a discount amount or rate has been stored; interpretation (currency vs fraction) depends on system conventions. To find records with an active discount, filter for non‑NULL and non‑zero values (recommended): PaymentDiscount IS NOT NULL AND PaymentDiscount <> 0. If your business rule guarantees only positive discounts, you can use PaymentDiscount > 0. Confirm with domain documentation whether 0.000 is used to mean explicit 'no discount' and whether negative values are valid before applying stricter filters.,
    ProductDiscount -- ProductDiscount is a nullable numeric field used to indicate whether a discount applies to the FreeTemplate row. NULL = no discount specified/not applicable. A populated DECIMAL value = a discount is defined (interpretation as currency vs percent must be confirmed with business rules). To treat a row as 'active' meaning a discount is in effect, filter for ProductDiscount IS NOT NULL AND ProductDiscount <> 0. If your business treats a zero value as meaningful (explicitly 'no discount') then use ProductDiscount IS NOT NULL to find rows where a discount was set, or ProductDiscount > 0 to find only positive discounts. Also validate whether negative values are allowed and prefer exact numeric comparisons (DECIMAL is precise); use COALESCE/ProductDiscount defaulting only when appropriate.,
    BundleDiscount -- BundleDiscount is not a boolean flag but a numeric indicator of a discount. NULL signals absence/unspecified discount; any populated number means an explicitly stored discount magnitude. To treat "active" as rows with an actual discount applied, filter for non-NULL values greater than zero (BundleDiscount IS NOT NULL AND BundleDiscount > 0). If your business defines "active" differently (for example, zero is considered active, or the column stores a percentage vs. absolute amount), adapt the condition accordingly and confirm the intended units/semantics with domain rules.,
    BudgetAmount -- BudgetAmount,
    OneTimeAmount -- OneTimeAmount,
    ExtraFeeAmount -- ExtraFeeAmount,
    Months -- Months,
    OneTimeAmountAfterDiscount -- This column indicates whether a one‑time charge (after discounts) exists and, if so, its amount. NULL should be interpreted as 'no value recorded / not applicable / unknown.' Any non‑NULL value indicates a defined post‑discount amount; consumers of the data should treat 0.00 as an explicit zero charge and decide whether that counts as 'active' for their use case. If you want rows that merely have an amount defined, filter with OneTimeAmountAfterDiscount IS NOT NULL. If you only want rows with a positive charge, use OneTimeAmountAfterDiscount > 0. If you want to treat zero as inactive, use OneTimeAmountAfterDiscount IS NOT NULL AND OneTimeAmountAfterDiscount <> 0.,
    ExtraFeeAmountAfterDiscount -- This column is a quantitative money field rather than a boolean flag. NULL typically means no fee is applicable or the amount hasn't been set/calculated. A populated money value is the extra fee remaining after discounts have been applied; interpreting that value depends on business rules (treat >0 as an active fee, =0 as no fee, negative as credit). To find rows with an active/recorded fee, filter for non-NULL and non-zero (or use > 0 if you only consider positive charges as active).,
    ActualSpentBudget -- ActualSpentBudget,
    ExtraFeeAmountAfterCC -- ExtraFeeAmountAfterCC,
    AlteredPPCAmountAfterCC -- AlteredPPCAmountAfterCC,
    IsInterruptedFromCRM -- IsInterruptedFromCRM is a NOT NULL BIT flag that marks whether a FreeTemplate was interrupted by CRM. Because it cannot be NULL under the schema, every row should contain either 0 (not interrupted) or 1 (interrupted). Treat rows with 0 as the active/normal templates and rows with 1 as interrupted/flagged templates. If you need defensive code against unexpected NULLs from upstream issues, use ISNULL(IsInterruptedFromCRM, 0) = 0 or filter out NULLs explicitly while investigating.,
    -- Measure: TotalState = SUM(State),
    -- Measure: TotalPrice = SUM(Price),
    -- Measure: TotalSalesTypeID = SUM(SalesTypeID),
    -- Measure: TotalQuantity = SUM(Quantity),
    -- Measure: TotalDiscount = SUM(Discount)
FROM dbo.FreeTemplate
WHERE
    State IS NULL AND IsROI IS NULL AND Discount IS NULL AND RulesDiscount IS NULL AND IsRenewed IS NULL AND IsRenewable IS NULL AND CancelledOn IS NULL AND CancelledBy IS NULL AND PaymentDiscount IS NULL AND ProductDiscount IS NULL AND BundleDiscount IS NULL AND OneTimeAmountAfterDiscount IS NULL AND ExtraFeeAmountAfterDiscount IS NULL AND IsInterruptedFromCRM IS NULL
    -- Filter for active records only
GO

-- Fact: FreeTemplateAdvertisement
-- Source: dbo.FreeTemplateAdvertisement
CREATE OR ALTER VIEW semantic.FreeTemplateAdvertisement AS
SELECT
    ID -- ID,
    FreeTemplateID -- FreeTemplateID,
    ProductID -- ProductID,
    VersionSectionID -- VersionSectionID,
    GeoAreaID -- GeoAreaID,
    HeadingID -- HeadingID,
    AdvBeginDate -- AdvBeginDate,
    AdvEndDate -- AdvEndDate,
    QA -- QA,
    CaseAttachmentID -- CaseAttachmentID,
    DMKey -- DMKey,
    DMTable -- DMTable,
    DMCharKey -- DMCharKey,
    Price -- Price,
    Discount -- Although requested as a status/flag analysis, Discount is a numeric money field rather than a true status indicator. Because it is non-nullable there is no NULL state to indicate unknown/inactive. Use Discount > 0 to find records with an active discount (adjust to >= 0 or include negatives only if your business rules treat zero or negative values as active).,
    RulesDiscount -- Because RulesDiscount is non-nullable, NULL does not signal 'inactive'. Instead a zero value normally means 'no discount' and any non-zero value means the record carries a rules discount. To select records with an applied discount use RulesDiscount <> 0; if your business definition of active requires only positive discounts use RulesDiscount > 0. Be cautious about negative values and handle them according to business rules.,
    ProductKitID -- ProductKitID,
    IsSubscriptionBased -- IsSubscriptionBased is a mandatory boolean flag that marks whether a FreeTemplateAdvertisement is subscription‑based. Because the column is NOT NULL, every row should contain either 0 (not subscription‑based) or 1 (subscription‑based). Use WHERE IsSubscriptionBased = 1 to retrieve subscription‑based records and WHERE IsSubscriptionBased = 0 for non‑subscription records. If NULLs are unexpectedly present, treat them as an unknown/unset state and investigate or normalize them with ISNULL/COALESCE as appropriate.,
    IsVerified -- IsVerified is a non-nullable boolean flag indicating whether the FreeTemplateAdvertisement has been verified. Because the column is NOT NULL, every row should have either 1 (verified) or 0 (not verified). To select records considered "active" in the sense of being verified, filter with IsVerified = 1. To select unverified records use IsVerified = 0. If you encounter NULLs despite the NOT NULL definition, investigate data corruption, ETL issues, or an out-of-date schema.,
    AlteredAmount -- AlteredAmount,
    DenyExtraction -- DenyExtraction,
    DesiredBeginDate -- DesiredBeginDate,
    DeactivationDate -- DeactivationDate,
    DeactivationReason -- DeactivationReason,
    CompletedChanges -- CompletedChanges,
    PaymentDiscount -- Although labeled here as a status/flag analysis, PaymentDiscount is a numeric field (not a boolean flag). NULL should be treated as 'no discount specified / not applicable'. A non-NULL value indicates a recorded discount amount or percentage. To identify records with an active/meaningful discount, filter for PaymentDiscount IS NOT NULL AND PaymentDiscount > 0. If your definition of active includes explicitly recorded zero discounts, use PaymentDiscount IS NOT NULL. Verify domain rules (currency vs percent, allowed negative values) with source documentation before applying business logic.,
    ProductDiscount -- ProductDiscount is a numeric status-like field indicating whether a discount exists. NULL means no discount information (not applicable or unknown). A populated DECIMAL means a discount value was supplied; interpret units from business rules (currency vs percent). To select records where a discount is actively applied, filter for non-NULL values greater than zero (ProductDiscount IS NOT NULL AND ProductDiscount > 0). If your domain treats 0 differently, adjust the condition accordingly (e.g., include = 0 if 0 should be considered active).,
    BundleDiscount -- BundleDiscount is primarily a numeric attribute, not a boolean flag. NULL indicates absence of a configured discount (or not applicable). A populated DECIMAL value conveys the discount amount; you must consult business metadata to know if the number is a percent, absolute currency, or stored in basis points. To treat records with an effective discount as "active", filter for non‑NULL and positive values (BundleDiscount IS NOT NULL AND BundleDiscount > 0). If your domain uses 0 as a sentinel for "no discount" but allows negative or zero values for other reasons, adjust the filter (for example use BundleDiscount IS NOT NULL AND BundleDiscount <> 0) after confirming business rules.,
    BudgetAmount -- BudgetAmount,
    OneTimeAmount -- OneTimeAmount,
    ExtraFeeAmount -- ExtraFeeAmount,
    Months -- Months,
    SendTo -- SendTo,
    SendOn -- SendOn,
    CanExport -- CanExport,
    RelatedFreeTemplateAdvertisementID -- This nullable integer likely functions as a self-referential foreign key to indicate that a row is linked to another advertisement. NULL typically denotes a primary/original or otherwise standalone record (often treated as the active/current item), while a populated value points to a related record (child/derivative/replacement). If you consider 'active' records to be the primary/unreplaced ones, filter using RelatedFreeTemplateAdvertisementID IS NULL. If your domain defines 'active' differently (for example, only linked records are active), use RelatedFreeTemplateAdvertisementID IS NOT NULL. Confirm intent against application logic or documentation before relying on this for business-critical filters.,
    OneTimeAmountAfterDiscount -- This nullable money column acts like a presence/amount indicator: NULL signals absence/unknown of a one-time discounted charge; a populated MONEY value gives the actual post-discount amount. If you treat "active" as having a defined one-time amount, filter with OneTimeAmountAfterDiscount IS NOT NULL. If you instead need only positive charges, use OneTimeAmountAfterDiscount > 0.,
    ExtraFeeAmountAfterDiscount -- Although this is a monetary column rather than a classic boolean status flag, NULL typically denotes 'no data / not applicable / not charged', while a populated MONEY value indicates an explicitly recorded post-discount extra fee. To treat rows with an actual fee (including credits) as 'active', filter for non-null and non-zero values. If you only care about positive charges, use ExtraFeeAmountAfterDiscount > 0 instead.,
    ActualSpentBudget -- ActualSpentBudget,
    ExtraFeeAmountAfterCC -- ExtraFeeAmountAfterCC,
    AlteredPPCAmountAfterCC -- AlteredPPCAmountAfterCC,
    -- Measure: RecordCount = COUNT(*),
    -- Measure: DistinctProductCount = COUNT(DISTINCT ProductID),
    -- Measure: DistinctVersionSectionCount = COUNT(DISTINCT VersionSectionID),
    -- Measure: DistinctGeoAreaCount = COUNT(DISTINCT GeoAreaID),
    -- Measure: DistinctHeadingCount = COUNT(DISTINCT HeadingID),
    -- Measure: DistinctDMKeyCount = COUNT(DISTINCT DMKey),
    -- Measure: TotalPrice = SUM(Price),
    -- Measure: AveragePrice = AVG(Price),
    -- Measure: MinPrice = MIN(Price),
    -- Measure: MaxPrice = MAX(Price),
    -- Measure: TotalDiscount = SUM(Discount),
    -- Measure: TotalRulesDiscount = SUM(RulesDiscount),
    -- Measure: TotalAlteredAmount = SUM(AlteredAmount),
    -- Measure: TotalPaymentDiscount = SUM(PaymentDiscount),
    -- Measure: TotalProductDiscount = SUM(ProductDiscount),
    -- Measure: TotalBundleDiscount = SUM(BundleDiscount),
    -- Measure: TotalDiscountsCombined = SUM(COALESCE(Discount,0) + COALESCE(RulesDiscount,0) + COALESCE(PaymentDiscount,0) + COALESCE(ProductDiscount,0) + COALESCE(BundleDiscount,0)),
    -- Measure: AverageDiscountToPriceRatio = AVG(CASE WHEN Price <> 0 THEN (COALESCE(Discount,0) / Price) ELSE NULL END),
    -- Measure: TotalBudgetAmount = SUM(BudgetAmount),
    -- Measure: TotalActualSpentBudget = SUM(ActualSpentBudget),
    -- Measure: AverageBudgetUtilization = AVG(CASE WHEN COALESCE(BudgetAmount,0) <> 0 THEN (ActualSpentBudget / BudgetAmount) ELSE NULL END),
    -- Measure: TotalOneTimeAmount = SUM(OneTimeAmount),
    -- Measure: TotalExtraFeeAmount = SUM(ExtraFeeAmount),
    -- Measure: TotalOneTimeAmountAfterDiscount = SUM(OneTimeAmountAfterDiscount),
    -- Measure: TotalExtraFeeAmountAfterDiscount = SUM(ExtraFeeAmountAfterDiscount),
    -- Measure: TotalExtraFeeAmountAfterCC = SUM(ExtraFeeAmountAfterCC),
    -- Measure: TotalAlteredPPCAmountAfterCC = SUM(AlteredPPCAmountAfterCC),
    -- Measure: SumMonths = SUM(Months),
    -- Measure: AverageMonths = AVG(Months),
    -- Measure: MinMonths = MIN(Months),
    -- Measure: MaxMonths = MAX(Months),
    -- Measure: SumCompletedChanges = SUM(CompletedChanges),
    -- Measure: AverageCompletedChanges = AVG(CompletedChanges),
    -- Measure: MinCompletedChanges = MIN(CompletedChanges),
    -- Measure: MaxCompletedChanges = MAX(CompletedChanges),
    -- Measure: CountWithDeactivationReason = SUM(CASE WHEN DeactivationReason IS NOT NULL THEN 1 ELSE 0 END),
    -- Measure: DistinctDeactivationReasonCount = COUNT(DISTINCT DeactivationReason),
    -- Measure: DistinctRelatedFreeTemplateAdvertisementCount = COUNT(DISTINCT RelatedFreeTemplateAdvertisementID)
FROM dbo.FreeTemplateAdvertisement
WHERE
    Discount IS NULL AND RulesDiscount IS NULL AND IsVerified IS NULL AND PaymentDiscount IS NULL AND ProductDiscount IS NULL AND BundleDiscount IS NULL AND RelatedFreeTemplateAdvertisementID IS NULL AND OneTimeAmountAfterDiscount IS NULL AND ExtraFeeAmountAfterDiscount IS NULL
    -- Filter for active records only
GO

-- Fact: BatchActionHistoryAnalysis
-- Source: dbo.BatchActionHistoryAnalysis
CREATE OR ALTER VIEW semantic.BatchActionHistoryAnalysis AS
SELECT
    ID -- ID,
    BatchID -- BatchID,
    EntityID -- EntityID,
    EntityType -- EntityType,
    Type -- Type,
    EntityData -- EntityData,
    -- Measure: RecordCount = COUNT(*),
    -- Measure: DistinctEntityCount = COUNT(DISTINCT EntityID),
    -- Measure: DistinctEntityTypeCount = COUNT(DISTINCT EntityType),
    -- Measure: DistinctTypeCount = COUNT(DISTINCT Type),
    -- Measure: AverageEntityID = AVG(EntityID),
    -- Measure: AverageEntityType = AVG(EntityType),
    -- Measure: AverageType = AVG(Type),
    -- Measure: MinEntityID = MIN(EntityID),
    -- Measure: MaxEntityID = MAX(EntityID),
    -- Measure: MinEntityType = MIN(EntityType),
    -- Measure: MaxEntityType = MAX(EntityType),
    -- Measure: MinType = MIN(Type),
    -- Measure: MaxType = MAX(Type)
FROM dbo.BatchActionHistoryAnalysis
GO

-- Fact: TaskTarget
-- Source: dbo.TaskTarget
CREATE OR ALTER VIEW semantic.TaskTarget AS
SELECT
    TaskID -- TaskID,
    TargetCaseID -- TargetCaseID,
    DMKey -- DMKey,
    DMTable -- DMTable,
    DMCharKey -- DMCharKey,
    -- Measure: RecordCount = COUNT(*),
    -- Measure: DistinctDMKeyCount = COUNT(DISTINCT DMKey),
    -- Measure: TotalDMKey = SUM(DMKey),
    -- Measure: AverageDMKey = AVG(DMKey),
    -- Measure: MinDMKey = MIN(DMKey),
    -- Measure: MaxDMKey = MAX(DMKey),
    -- Measure: MissingDMKeyCount = SUM(CASE WHEN DMKey IS NULL THEN 1 ELSE 0 END)
FROM dbo.TaskTarget
GO

-- Fact: BatchActionHistory
-- Source: dbo.BatchActionHistory
CREATE OR ALTER VIEW semantic.BatchActionHistory AS
SELECT
    ID -- ID,
    BatchID -- BatchID,
    BatchActionType -- BatchActionType,
    TimeStamp -- TimeStamp,
    RowsAffected -- RowsAffected,
    Text -- Text,
    Status -- Because Status is NOT NULL, there is no NULL state — every record has a status code. Populated values are encoded integers whose meanings are defined by the application (common patterns include 0=New,1=Active/Completed,2=Failed,3=Cancelled, etc.). To select active records you must use the integer value(s) that the application designates as 'active' (example filter shown; consult documentation or a status lookup table to determine the correct code(s)).,
    Type -- Type,
    -- Measure: ActionCount = COUNT(*),
    -- Measure: TotalRowsAffected = SUM(RowsAffected),
    -- Measure: AverageRowsAffected = AVG(RowsAffected),
    -- Measure: MinRowsAffected = MIN(RowsAffected),
    -- Measure: MaxRowsAffected = MAX(RowsAffected),
    -- Measure: ActionsWithRowsAffectedCount = SUM(CASE WHEN RowsAffected > 0 THEN 1 ELSE 0 END),
    -- Measure: PercentActionsWithRowsAffected = SUM(CASE WHEN RowsAffected > 0 THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(*), 0),
    -- Measure: DistinctBatchActionTypes = COUNT(DISTINCT BatchActionType),
    -- Measure: MinBatchActionTypeCode = MIN(BatchActionType),
    -- Measure: MaxBatchActionTypeCode = MAX(BatchActionType),
    -- Measure: AverageBatchActionTypeCode = AVG(BatchActionType),
    -- Measure: DistinctStatuses = COUNT(DISTINCT Status),
    -- Measure: MinStatusCode = MIN(Status),
    -- Measure: MaxStatusCode = MAX(Status),
    -- Measure: AverageStatusCode = AVG(Status),
    -- Measure: DistinctTypes = COUNT(DISTINCT Type),
    -- Measure: MinTypeCode = MIN(Type),
    -- Measure: MaxTypeCode = MAX(Type),
    -- Measure: AverageTypeCode = AVG(Type)
FROM dbo.BatchActionHistory
WHERE
    Status IS NULL
    -- Filter for active records only
GO

-- Fact: TicketCommunication
-- Source: dbo.TicketCommunication
CREATE OR ALTER VIEW semantic.TicketCommunication AS
SELECT
    ID -- ID,
    TicketID -- TicketID,
    CommunicationText -- CommunicationText,
    -- Measure: CommunicationCount = COUNT(*),
    -- Measure: DistinctTicketCount = COUNT(DISTINCT TicketID),
    -- Measure: CommunicationsPerTicket = COUNT(*) * 1.0 / NULLIF(COUNT(DISTINCT TicketID), 0),
    -- Measure: MinTicketID = MIN(TicketID),
    -- Measure: MaxTicketID = MAX(TicketID),
    -- Measure: AverageTicketID = AVG(TicketID)
FROM dbo.TicketCommunication
GO

-- Fact: KeywordMapping
-- Source: MigrationSteps.KeywordMapping
CREATE OR ALTER VIEW semantic.KeywordMapping AS
SELECT
    ActivityID -- ActivityID,
    KeywordID -- KeywordID,
    NewAttributeValueID -- NewAttributeValueID,
    -- Measure: TotalRecords = COUNT(*),
    -- Measure: DistinctActivityCount = COUNT(DISTINCT ActivityID),
    -- Measure: DistinctKeywordCount = COUNT(DISTINCT KeywordID),
    -- Measure: DistinctNewAttributeValueCount = COUNT(DISTINCT NewAttributeValueID),
    -- Measure: RecordsWithNewAttributeValue = COUNT(NewAttributeValueID),
    -- Measure: MinActivityID = MIN(ActivityID),
    -- Measure: MaxActivityID = MAX(ActivityID),
    -- Measure: AverageActivityID = AVG(ActivityID),
    -- Measure: MinKeywordID = MIN(KeywordID),
    -- Measure: MaxKeywordID = MAX(KeywordID),
    -- Measure: AverageKeywordID = AVG(KeywordID),
    -- Measure: MinNewAttributeValueID = MIN(NewAttributeValueID),
    -- Measure: MaxNewAttributeValueID = MAX(NewAttributeValueID),
    -- Measure: AverageNewAttributeValueID = AVG(NewAttributeValueID)
FROM MigrationSteps.KeywordMapping
GO

-- Fact: Step1
-- Source: MigrationSteps.Step1
CREATE OR ALTER VIEW semantic.Step1 AS
SELECT
    ActivityID -- ActivityID,
    AttributeValueID -- AttributeValueID,
    -- Measure: RecordCount = COUNT(*),
    -- Measure: SumActivityID = SUM(ActivityID),
    -- Measure: AverageActivityID = AVG(ActivityID),
    -- Measure: MinActivityID = MIN(ActivityID),
    -- Measure: MaxActivityID = MAX(ActivityID),
    -- Measure: CountActivityID = COUNT(ActivityID),
    -- Measure: DistinctActivityIDCount = COUNT(DISTINCT ActivityID),
    -- Measure: SumAttributeValueID = SUM(AttributeValueID),
    -- Measure: AverageAttributeValueID = AVG(AttributeValueID),
    -- Measure: MinAttributeValueID = MIN(AttributeValueID),
    -- Measure: MaxAttributeValueID = MAX(AttributeValueID),
    -- Measure: CountAttributeValueID = COUNT(AttributeValueID),
    -- Measure: DistinctAttributeValueIDCount = COUNT(DISTINCT AttributeValueID)
FROM MigrationSteps.Step1
GO

-- Dimension: TaxCode
-- Source: dbo.TaxCode
CREATE OR ALTER VIEW semantic.TaxCode AS
SELECT
    ID -- ID,
    CountryID -- CountryID,
    TaxCode -- TaxCode,
    DMKey -- DMKey,
    DMTable -- DMTable,
    DMCharKey -- DMCharKey
FROM dbo.TaxCode
GO

-- Dimension: DEV12867ClassificationBackup
-- Source: HEADQUARTERS\cstraitchof.DEV12867ClassificationBackup
CREATE OR ALTER VIEW semantic.DEV12867ClassificationBackup AS
SELECT
    ID -- ID,
    BusinessPointID -- BusinessPointID,
    ClassifierID -- ClassifierID,
    ClassifierNodeID -- ClassifierNodeID,
    DMKey -- DMKey,
    DMTable -- DMTable,
    DMCharKey -- DMCharKey,
    BpCode -- BpCode
FROM HEADQUARTERS\cstraitchof.DEV12867ClassificationBackup
GO

-- Dimension: devops11385_InvoicePaymentMethod_NULL_PaymentMonth
-- Source: HEADQUARTERS\spitsaris.devops11385_InvoicePaymentMethod_NULL_PaymentMonth
CREATE OR ALTER VIEW semantic.devops11385_InvoicePaymentMethod_NULL_PaymentMonth AS
SELECT
    ID -- ID,
    PaymentMethodTypeID -- PaymentMethodTypeID,
    Installments -- Installments,
    Interval -- Interval,
    IsExchangeSale -- IsExchangeSale is a non-nullable boolean status flag indicating whether the invoice/payment method is an exchange sale. Because the column is NOT NULL, absence of a value is an error condition. Interpret populated values as 1 = exchange sale (active for this status) and 0 = not an exchange sale. To retrieve exchange-sale records use WHERE IsExchangeSale = 1; to retrieve non-exchange use WHERE IsExchangeSale = 0. If sample values are unavailable, validate this interpretation against business rules or lookups (SELECT DISTINCT IsExchangeSale) to confirm actual stored values.,
    PaymentPhoneAccount -- PaymentPhoneAccount,
    PaymentPhoneID -- PaymentPhoneID,
    PaymentCreditTransactionCode -- PaymentCreditTransactionCode,
    BillingEmailID -- BillingEmailID,
    NumberOfInvoices -- NumberOfInvoices,
    UserSpecifiedPaymentMonth -- UserSpecifiedPaymentMonth,
    BankID -- BankID,
    CheckEndDate -- CheckEndDate,
    CheckPublisherIsSame -- CheckPublisherIsSame is a nullable boolean flag indicating whether the publisher was verified as being the same. NULL = no decision/unknown/not checked. A stored 1 means confirmed 'same'; 0 means 'not same'. To select records where the flag is actively true, filter for value = 1; using ISNULL(CheckPublisherIsSame, 0) = 1 treats NULL as false and safely returns only confirmed-true rows.,
    CheckPublisher -- Treat CheckPublisher as a flag that is NULL until a check/publish event is recorded. When populated it signals the record was checked/published and gives the publisher identifier. Beware of empty strings or sentinel text ('N/A', 'UNKNOWN', 'None', etc.); include a trim and exclusion of common sentinel values if your dataset uses them. Example safe filter for 'active' records: WHERE CheckPublisher IS NOT NULL AND LTRIM(RTRIM(CheckPublisher)) <> '' (and optionally AND CheckPublisher NOT IN ('N/A','UNKNOWN','None')). The Greek_CI_AS collation is case-insensitive, so you do not need additional case handling for textual sentinel comparisons.,
    CheckPublisherTaxCode -- This NVARCHAR(20) nullable column functions as a status/value indicator: NULL means no data or no check performed (unknown/missing/not applicable). A populated value means the row has a recorded outcome or tax identifier; the payload could be the tax code itself or a status token. To treat "active" as "has been processed / has a recorded value", filter for non-NULL, non-empty strings using the provided condition. If your system uses explicit positive/negative tokens (e.g. 'Y'/'N', 'VALID'/'INVALID'), refine the filter to match the positive token(s) (for example: CheckPublisherTaxCode = 'Y' OR CheckPublisherTaxCode = 'VALID'). Verify actual stored values before finalizing any rules.,
    CheckNumber -- CheckNumber,
    CreditCardTypeID -- CreditCardTypeID,
    CreditCardNumber -- CreditCardNumber,
    CreditCardEndDate -- CreditCardEndDate,
    CreditCardOwner -- CreditCardOwner,
    CreditCardInstallments -- CreditCardInstallments,
    BlockSeries -- BlockSeries,
    BlockReceiptNumber -- BlockReceiptNumber,
    DMKey -- DMKey,
    DMTable -- DMTable,
    DMCharKey -- DMCharKey,
    SystemReceiptNumber -- SystemReceiptNumber,
    SalesCollectionID -- SalesCollectionID,
    PaymentMonth -- PaymentMonth,
    CheckPublishDate -- This DATETIME column is acting as a completion/publish flag: NULL denotes that publishing has not yet occurred (the record is still pending or active for work), while a populated datetime marks the moment the check was published (completed). To select records that are still active/pending use the NULL check (CheckPublishDate IS NULL). If you instead need records that have been published/closed, filter with CheckPublishDate IS NOT NULL.,
    BankDepositNumber -- BankDepositNumber,
    BankDepositDate -- BankDepositDate,
    CollectionDate -- CollectionDate,
    AdvancedPolicyAmount -- AdvancedPolicyAmount,
    InvoicePolicyID -- InvoicePolicyID,
    PreInvoice -- PreInvoice
FROM HEADQUARTERS\spitsaris.devops11385_InvoicePaymentMethod_NULL_PaymentMonth
GO

-- Dimension: ViewPrefectureBPs
-- Source: HEADQUARTERS\spitsaris.ViewPrefectureBPs
CREATE OR ALTER VIEW semantic.ViewPrefectureBPs AS
SELECT
    ActiveFlag -- ActiveFlag is a required boolean status indicator for the record. Because the column is non‑nullable, every row must explicitly indicate active (1) or inactive (0). To retrieve active records, filter for ActiveFlag = 1. To retrieve inactive records, filter for ActiveFlag = 0. If you need to handle legacy or unexpected values, validate data quality or check for constraints/defaults in the schema.,
    AreaID -- AreaID,
    AreaName -- AreaName,
    BPcount -- BPcount
FROM HEADQUARTERS\spitsaris.ViewPrefectureBPs
GO

-- Dimension: AttributeMapping
-- Source: MigrationSteps.AttributeMapping
CREATE OR ALTER VIEW semantic.AttributeMapping AS
SELECT
    ActivityID -- ActivityID,
    OldAttributeID -- OldAttributeID,
    NewAttributeID -- NewAttributeID
FROM MigrationSteps.AttributeMapping
GO

-- Dimension: ReplaceKeywords
-- Source: MigrationSteps.ReplaceKeywords
CREATE OR ALTER VIEW semantic.ReplaceKeywords AS
SELECT
    KeywordID -- KeywordID,
    ReplaceWithKeywordID -- ReplaceWithKeywordID
FROM MigrationSteps.ReplaceKeywords
GO

-- Dimension: Step2
-- Source: MigrationSteps.Step2
CREATE OR ALTER VIEW semantic.Step2 AS
SELECT
    Activity_ID -- Activity_ID,
    NameGr -- NameGr,
    NameEn -- NameEn
FROM MigrationSteps.Step2
GO

-- Dimension: MergeMapping
-- Source: MigrationSteps.MergeMapping
CREATE OR ALTER VIEW semantic.MergeMapping AS
SELECT
    SourceID -- SourceID,
    TargetID -- TargetID
FROM MigrationSteps.MergeMapping
GO

-- Dimension: ReplaceAV
-- Source: MigrationSteps.ReplaceAV
CREATE OR ALTER VIEW semantic.ReplaceAV AS
SELECT
    Initial_OldAttributeValueId -- Initial_OldAttributeValueId,
    ReplaceWithAttributeValueId -- ReplaceWithAttributeValueId
FROM MigrationSteps.ReplaceAV
GO

-- Metric: ContentAttributeValue RecordCount
-- Purpose: Track RecordCount
-- Logic: COUNT(*)
CREATE OR ALTER VIEW semantic.Metric_ContentAttributeValue_RecordCount AS
-- TODO: Implement metric aggregation
-- Inputs: ContentAttributeValue.RecordCount
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: ContentAttributeValue TotalSeqNo
-- Purpose: Track TotalSeqNo
-- Logic: SUM(SeqNo)
CREATE OR ALTER VIEW semantic.Metric_ContentAttributeValue_TotalSeqNo AS
-- TODO: Implement metric aggregation
-- Inputs: ContentAttributeValue.TotalSeqNo
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: ContentAttributeValue AverageSeqNo
-- Purpose: Track AverageSeqNo
-- Logic: AVG(SeqNo)
CREATE OR ALTER VIEW semantic.Metric_ContentAttributeValue_AverageSeqNo AS
-- TODO: Implement metric aggregation
-- Inputs: ContentAttributeValue.AverageSeqNo
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: FreeTemplateAttributeValue RecordCount
-- Purpose: Track RecordCount
-- Logic: COUNT(*)
CREATE OR ALTER VIEW semantic.Metric_FreeTemplateAttributeValue_RecordCount AS
-- TODO: Implement metric aggregation
-- Inputs: FreeTemplateAttributeValue.RecordCount
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: FreeTemplateAttributeValue DistinctDMKeyCount
-- Purpose: Track DistinctDMKeyCount
-- Logic: COUNT(DISTINCT DMKey)
CREATE OR ALTER VIEW semantic.Metric_FreeTemplateAttributeValue_DistinctDMKeyCount AS
-- TODO: Implement metric aggregation
-- Inputs: FreeTemplateAttributeValue.DistinctDMKeyCount
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: FreeTemplateAttributeValue SumSeqNo
-- Purpose: Track SumSeqNo
-- Logic: SUM(SeqNo)
CREATE OR ALTER VIEW semantic.Metric_FreeTemplateAttributeValue_SumSeqNo AS
-- TODO: Implement metric aggregation
-- Inputs: FreeTemplateAttributeValue.SumSeqNo
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: TaskLog TaskLogCount
-- Purpose: Track TaskLogCount
-- Logic: COUNT(*)
CREATE OR ALTER VIEW semantic.Metric_TaskLog_TaskLogCount AS
-- TODO: Implement metric aggregation
-- Inputs: TaskLog.TaskLogCount
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: TaskLog DistinctDMKeyCount
-- Purpose: Track DistinctDMKeyCount
-- Logic: COUNT(DISTINCT DMKey)
CREATE OR ALTER VIEW semantic.Metric_TaskLog_DistinctDMKeyCount AS
-- TODO: Implement metric aggregation
-- Inputs: TaskLog.DistinctDMKeyCount
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: TaskLog TotalDMKey
-- Purpose: Track TotalDMKey
-- Logic: SUM(DMKey)
CREATE OR ALTER VIEW semantic.Metric_TaskLog_TotalDMKey AS
-- TODO: Implement metric aggregation
-- Inputs: TaskLog.TotalDMKey
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: TaskAssignment AssignmentCount
-- Purpose: Track AssignmentCount
-- Logic: COUNT(*)
CREATE OR ALTER VIEW semantic.Metric_TaskAssignment_AssignmentCount AS
-- TODO: Implement metric aggregation
-- Inputs: TaskAssignment.AssignmentCount
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: TaskAssignment DistinctDMKeyCount
-- Purpose: Track DistinctDMKeyCount
-- Logic: COUNT(DISTINCT DMKey)
CREATE OR ALTER VIEW semantic.Metric_TaskAssignment_DistinctDMKeyCount AS
-- TODO: Implement metric aggregation
-- Inputs: TaskAssignment.DistinctDMKeyCount
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: TaskAssignment DistinctAssignmentTypeCount
-- Purpose: Track DistinctAssignmentTypeCount
-- Logic: COUNT(DISTINCT AssignmentType)
CREATE OR ALTER VIEW semantic.Metric_TaskAssignment_DistinctAssignmentTypeCount AS
-- TODO: Implement metric aggregation
-- Inputs: TaskAssignment.DistinctAssignmentTypeCount
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: TimeLogger RecordCount
-- Purpose: Track RecordCount
-- Logic: COUNT(ID)
CREATE OR ALTER VIEW semantic.Metric_TimeLogger_RecordCount AS
-- TODO: Implement metric aggregation
-- Inputs: TimeLogger.RecordCount
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: TimeLogger DistinctIDCount
-- Purpose: Track DistinctIDCount
-- Logic: COUNT(DISTINCT ID)
CREATE OR ALTER VIEW semantic.Metric_TimeLogger_DistinctIDCount AS
-- TODO: Implement metric aggregation
-- Inputs: TimeLogger.DistinctIDCount
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: TimeLogger MinID
-- Purpose: Track MinID
-- Logic: MIN(ID)
CREATE OR ALTER VIEW semantic.Metric_TimeLogger_MinID AS
-- TODO: Implement metric aggregation
-- Inputs: TimeLogger.MinID
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: TargetGroupItem RecordCount
-- Purpose: Track RecordCount
-- Logic: COUNT(*)
CREATE OR ALTER VIEW semantic.Metric_TargetGroupItem_RecordCount AS
-- TODO: Implement metric aggregation
-- Inputs: TargetGroupItem.RecordCount
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: TargetGroupItem MainHeadingID_Count
-- Purpose: Track MainHeadingID_Count
-- Logic: COUNT(MainHeadingID)
CREATE OR ALTER VIEW semantic.Metric_TargetGroupItem_MainHeadingID_Count AS
-- TODO: Implement metric aggregation
-- Inputs: TargetGroupItem.MainHeadingID_Count
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: TargetGroupItem DMKey_Count
-- Purpose: Track DMKey_Count
-- Logic: COUNT(DMKey)
CREATE OR ALTER VIEW semantic.Metric_TargetGroupItem_DMKey_Count AS
-- TODO: Implement metric aggregation
-- Inputs: TargetGroupItem.DMKey_Count
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: PricelistItemDetail TotalUnitAmount
-- Purpose: Track TotalUnitAmount
-- Logic: SUM(UnitAmount)
CREATE OR ALTER VIEW semantic.Metric_PricelistItemDetail_TotalUnitAmount AS
-- TODO: Implement metric aggregation
-- Inputs: PricelistItemDetail.TotalUnitAmount
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: PricelistItemDetail AverageUnitAmount
-- Purpose: Track AverageUnitAmount
-- Logic: AVG(UnitAmount)
CREATE OR ALTER VIEW semantic.Metric_PricelistItemDetail_AverageUnitAmount AS
-- TODO: Implement metric aggregation
-- Inputs: PricelistItemDetail.AverageUnitAmount
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: PricelistItemDetail MinimumUnitAmount
-- Purpose: Track MinimumUnitAmount
-- Logic: MIN(UnitAmount)
CREATE OR ALTER VIEW semantic.Metric_PricelistItemDetail_MinimumUnitAmount AS
-- TODO: Implement metric aggregation
-- Inputs: PricelistItemDetail.MinimumUnitAmount
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: AuditLog RecordCount
-- Purpose: Track RecordCount
-- Logic: COUNT(*)
CREATE OR ALTER VIEW semantic.Metric_AuditLog_RecordCount AS
-- TODO: Implement metric aggregation
-- Inputs: AuditLog.RecordCount
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: AuditLog DistinctEntityCount
-- Purpose: Track DistinctEntityCount
-- Logic: COUNT(DISTINCT EntityID)
CREATE OR ALTER VIEW semantic.Metric_AuditLog_DistinctEntityCount AS
-- TODO: Implement metric aggregation
-- Inputs: AuditLog.DistinctEntityCount
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: AuditLog NonNullEntityCount
-- Purpose: Track NonNullEntityCount
-- Logic: COUNT(EntityID)
CREATE OR ALTER VIEW semantic.Metric_AuditLog_NonNullEntityCount AS
-- TODO: Implement metric aggregation
-- Inputs: AuditLog.NonNullEntityCount
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: BusinessPointComment TotalComments
-- Purpose: Track TotalComments
-- Logic: COUNT(*)
CREATE OR ALTER VIEW semantic.Metric_BusinessPointComment_TotalComments AS
-- TODO: Implement metric aggregation
-- Inputs: BusinessPointComment.TotalComments
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: BusinessPointComment DistinctUsers
-- Purpose: Track DistinctUsers
-- Logic: COUNT(DISTINCT UserID)
CREATE OR ALTER VIEW semantic.Metric_BusinessPointComment_DistinctUsers AS
-- TODO: Implement metric aggregation
-- Inputs: BusinessPointComment.DistinctUsers
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: BusinessPointComment DistinctBusinessPoints
-- Purpose: Track DistinctBusinessPoints
-- Logic: COUNT(DISTINCT BusinessPointID)
CREATE OR ALTER VIEW semantic.Metric_BusinessPointComment_DistinctBusinessPoints AS
-- TODO: Implement metric aggregation
-- Inputs: BusinessPointComment.DistinctBusinessPoints
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: DialerTask TaskCount
-- Purpose: Track TaskCount
-- Logic: COUNT([ID])
CREATE OR ALTER VIEW semantic.Metric_DialerTask_TaskCount AS
-- TODO: Implement metric aggregation
-- Inputs: DialerTask.TaskCount
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: DialerTask DistinctTaskIDs
-- Purpose: Track DistinctTaskIDs
-- Logic: COUNT(DISTINCT [ID])
CREATE OR ALTER VIEW semantic.Metric_DialerTask_DistinctTaskIDs AS
-- TODO: Implement metric aggregation
-- Inputs: DialerTask.DistinctTaskIDs
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: DialerTask MinTaskID
-- Purpose: Track MinTaskID
-- Logic: MIN([ID])
CREATE OR ALTER VIEW semantic.Metric_DialerTask_MinTaskID AS
-- TODO: Implement metric aggregation
-- Inputs: DialerTask.MinTaskID
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: TargetGroupItemPeriod TotalRecords
-- Purpose: Track TotalRecords
-- Logic: COUNT(*)
CREATE OR ALTER VIEW semantic.Metric_TargetGroupItemPeriod_TotalRecords AS
-- TODO: Implement metric aggregation
-- Inputs: TargetGroupItemPeriod.TotalRecords
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: TargetGroupItemPeriod DMKeyNonNullCount
-- Purpose: Track DMKeyNonNullCount
-- Logic: COUNT(DMKey)
CREATE OR ALTER VIEW semantic.Metric_TargetGroupItemPeriod_DMKeyNonNullCount AS
-- TODO: Implement metric aggregation
-- Inputs: TargetGroupItemPeriod.DMKeyNonNullCount
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: TargetGroupItemPeriod DistinctTargetGroups
-- Purpose: Track DistinctTargetGroups
-- Logic: COUNT(DISTINCT DMKey)
CREATE OR ALTER VIEW semantic.Metric_TargetGroupItemPeriod_DistinctTargetGroups AS
-- TODO: Implement metric aggregation
-- Inputs: TargetGroupItemPeriod.DistinctTargetGroups
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: Advertisement TotalProductID
-- Purpose: Track TotalProductID
-- Logic: SUM(ProductID)
CREATE OR ALTER VIEW semantic.Metric_Advertisement_TotalProductID AS
-- TODO: Implement metric aggregation
-- Inputs: Advertisement.TotalProductID
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: Advertisement TotalVersionSectionID
-- Purpose: Track TotalVersionSectionID
-- Logic: SUM(VersionSectionID)
CREATE OR ALTER VIEW semantic.Metric_Advertisement_TotalVersionSectionID AS
-- TODO: Implement metric aggregation
-- Inputs: Advertisement.TotalVersionSectionID
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: Advertisement TotalGeoAreaID
-- Purpose: Track TotalGeoAreaID
-- Logic: SUM(GeoAreaID)
CREATE OR ALTER VIEW semantic.Metric_Advertisement_TotalGeoAreaID AS
-- TODO: Implement metric aggregation
-- Inputs: Advertisement.TotalGeoAreaID
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: CampaignNewLossCustomer TotalRecords
-- Purpose: Track TotalRecords
-- Logic: COUNT(*)
CREATE OR ALTER VIEW semantic.Metric_CampaignNewLossCustomer_TotalRecords AS
-- TODO: Implement metric aggregation
-- Inputs: CampaignNewLossCustomer.TotalRecords
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: CampaignNewLossCustomer DistinctRuleCount
-- Purpose: Track DistinctRuleCount
-- Logic: COUNT(DISTINCT RuleID)
CREATE OR ALTER VIEW semantic.Metric_CampaignNewLossCustomer_DistinctRuleCount AS
-- TODO: Implement metric aggregation
-- Inputs: CampaignNewLossCustomer.DistinctRuleCount
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: CampaignNewLossCustomer MinRuleID
-- Purpose: Track MinRuleID
-- Logic: MIN(RuleID)
CREATE OR ALTER VIEW semantic.Metric_CampaignNewLossCustomer_MinRuleID AS
-- TODO: Implement metric aggregation
-- Inputs: CampaignNewLossCustomer.MinRuleID
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: Classification RecordCount
-- Purpose: Track RecordCount
-- Logic: COUNT(*)
CREATE OR ALTER VIEW semantic.Metric_Classification_RecordCount AS
-- TODO: Implement metric aggregation
-- Inputs: Classification.RecordCount
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: Classification UniqueDMKeyCount
-- Purpose: Track UniqueDMKeyCount
-- Logic: COUNT(DISTINCT DMKey)
CREATE OR ALTER VIEW semantic.Metric_Classification_UniqueDMKeyCount AS
-- TODO: Implement metric aggregation
-- Inputs: Classification.UniqueDMKeyCount
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: Classification TotalDMKey
-- Purpose: Track TotalDMKey
-- Logic: SUM(DMKey)
CREATE OR ALTER VIEW semantic.Metric_Classification_TotalDMKey AS
-- TODO: Implement metric aggregation
-- Inputs: Classification.TotalDMKey
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: ContractProduct TotalState
-- Purpose: Track TotalState
-- Logic: SUM(State)
CREATE OR ALTER VIEW semantic.Metric_ContractProduct_TotalState AS
-- TODO: Implement metric aggregation
-- Inputs: ContractProduct.TotalState
-- Constraints: State = 1 -- (use the integer code that represents 'active' in your environment; alternatively use a lookup: EXISTS (SELECT 1 FROM dbo.ContractProductState s WHERE s.Code = dbo.ContractProduct.State AND s.IsActive = 1))
SELECT 1 AS PlaceholderMetric
GO

-- Metric: ContractProduct TotalPrice
-- Purpose: Track TotalPrice
-- Logic: SUM(Price)
CREATE OR ALTER VIEW semantic.Metric_ContractProduct_TotalPrice AS
-- TODO: Implement metric aggregation
-- Inputs: ContractProduct.TotalPrice
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: ContractProduct TotalSalesTypeID
-- Purpose: Track TotalSalesTypeID
-- Logic: SUM(SalesTypeID)
CREATE OR ALTER VIEW semantic.Metric_ContractProduct_TotalSalesTypeID AS
-- TODO: Implement metric aggregation
-- Inputs: ContractProduct.TotalSalesTypeID
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: CaseComment TotalComments
-- Purpose: Track TotalComments
-- Logic: COUNT(*)
CREATE OR ALTER VIEW semantic.Metric_CaseComment_TotalComments AS
-- TODO: Implement metric aggregation
-- Inputs: CaseComment.TotalComments
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: CaseComment CommentType_Count
-- Purpose: Track CommentType_Count
-- Logic: COUNT(CommentType)
CREATE OR ALTER VIEW semantic.Metric_CaseComment_CommentType_Count AS
-- TODO: Implement metric aggregation
-- Inputs: CaseComment.CommentType_Count
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: CaseComment CommentType_DistinctCount
-- Purpose: Track CommentType_DistinctCount
-- Logic: COUNT(DISTINCT CommentType)
CREATE OR ALTER VIEW semantic.Metric_CaseComment_CommentType_DistinctCount AS
-- TODO: Implement metric aggregation
-- Inputs: CaseComment.CommentType_DistinctCount
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: CampaignScoreHistoryDetail RecordCount
-- Purpose: Track RecordCount
-- Logic: COUNT(ID)
CREATE OR ALTER VIEW semantic.Metric_CampaignScoreHistoryDetail_RecordCount AS
-- TODO: Implement metric aggregation
-- Inputs: CampaignScoreHistoryDetail.RecordCount
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: CampaignScoreHistoryDetail DistinctCampaignCount
-- Purpose: Track DistinctCampaignCount
-- Logic: COUNT(DISTINCT CampaignID)
CREATE OR ALTER VIEW semantic.Metric_CampaignScoreHistoryDetail_DistinctCampaignCount AS
-- TODO: Implement metric aggregation
-- Inputs: CampaignScoreHistoryDetail.DistinctCampaignCount
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: CampaignScoreHistoryDetail DistinctBusinessPointCount
-- Purpose: Track DistinctBusinessPointCount
-- Logic: COUNT(DISTINCT BusinessPointID)
CREATE OR ALTER VIEW semantic.Metric_CampaignScoreHistoryDetail_DistinctBusinessPointCount AS
-- TODO: Implement metric aggregation
-- Inputs: CampaignScoreHistoryDetail.DistinctBusinessPointCount
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: PointRelationship RecordCount
-- Purpose: Track RecordCount
-- Logic: COUNT(*)
CREATE OR ALTER VIEW semantic.Metric_PointRelationship_RecordCount AS
-- TODO: Implement metric aggregation
-- Inputs: PointRelationship.RecordCount
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: PointRelationship DistinctDMKeyCount
-- Purpose: Track DistinctDMKeyCount
-- Logic: COUNT(DISTINCT DMKey)
CREATE OR ALTER VIEW semantic.Metric_PointRelationship_DistinctDMKeyCount AS
-- TODO: Implement metric aggregation
-- Inputs: PointRelationship.DistinctDMKeyCount
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: PointRelationship DistinctTypeCount
-- Purpose: Track DistinctTypeCount
-- Logic: COUNT(DISTINCT Type)
CREATE OR ALTER VIEW semantic.Metric_PointRelationship_DistinctTypeCount AS
-- TODO: Implement metric aggregation
-- Inputs: PointRelationship.DistinctTypeCount
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: BusinessPointConfirmationHistory CountConfirmations
-- Purpose: Track CountConfirmations
-- Logic: COUNT(*)
CREATE OR ALTER VIEW semantic.Metric_BusinessPointConfirmationHistory_CountConfirmations AS
-- TODO: Implement metric aggregation
-- Inputs: BusinessPointConfirmationHistory.CountConfirmations
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: BusinessPointConfirmationHistory DistinctBusinessPointCount
-- Purpose: Track DistinctBusinessPointCount
-- Logic: COUNT(DISTINCT BusinessPointIdentificationID)
CREATE OR ALTER VIEW semantic.Metric_BusinessPointConfirmationHistory_DistinctBusinessPointCount AS
-- TODO: Implement metric aggregation
-- Inputs: BusinessPointConfirmationHistory.DistinctBusinessPointCount
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: BusinessPointConfirmationHistory DistinctPerformerCount
-- Purpose: Track DistinctPerformerCount
-- Logic: COUNT(DISTINCT PerformedByID)
CREATE OR ALTER VIEW semantic.Metric_BusinessPointConfirmationHistory_DistinctPerformerCount AS
-- TODO: Implement metric aggregation
-- Inputs: BusinessPointConfirmationHistory.DistinctPerformerCount
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: UserLog RecordCount
-- Purpose: Track RecordCount
-- Logic: COUNT(*)
CREATE OR ALTER VIEW semantic.Metric_UserLog_RecordCount AS
-- TODO: Implement metric aggregation
-- Inputs: UserLog.RecordCount
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: UserLog ActionNonNullCount
-- Purpose: Track ActionNonNullCount
-- Logic: COUNT(Action)
CREATE OR ALTER VIEW semantic.Metric_UserLog_ActionNonNullCount AS
-- TODO: Implement metric aggregation
-- Inputs: UserLog.ActionNonNullCount
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: UserLog DistinctActionCount
-- Purpose: Track DistinctActionCount
-- Logic: COUNT(DISTINCT Action)
CREATE OR ALTER VIEW semantic.Metric_UserLog_DistinctActionCount AS
-- TODO: Implement metric aggregation
-- Inputs: UserLog.DistinctActionCount
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: Ticket TicketCount
-- Purpose: Track TicketCount
-- Logic: COUNT(DMKey)
CREATE OR ALTER VIEW semantic.Metric_Ticket_TicketCount AS
-- TODO: Implement metric aggregation
-- Inputs: Ticket.TicketCount
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: Ticket DistinctResultCount
-- Purpose: Track DistinctResultCount
-- Logic: COUNT(DISTINCT Result)
CREATE OR ALTER VIEW semantic.Metric_Ticket_DistinctResultCount AS
-- TODO: Implement metric aggregation
-- Inputs: Ticket.DistinctResultCount
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: Ticket ResultMin
-- Purpose: Track ResultMin
-- Logic: MIN(Result)
CREATE OR ALTER VIEW semantic.Metric_Ticket_ResultMin AS
-- TODO: Implement metric aggregation
-- Inputs: Ticket.ResultMin
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: TicketResult TotalResult
-- Purpose: Track TotalResult
-- Logic: SUM(Result)
CREATE OR ALTER VIEW semantic.Metric_TicketResult_TotalResult AS
-- TODO: Implement metric aggregation
-- Inputs: TicketResult.TotalResult
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: TicketResult AverageResult
-- Purpose: Track AverageResult
-- Logic: AVG(Result)
CREATE OR ALTER VIEW semantic.Metric_TicketResult_AverageResult AS
-- TODO: Implement metric aggregation
-- Inputs: TicketResult.AverageResult
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: TicketResult MinResult
-- Purpose: Track MinResult
-- Logic: MIN(Result)
CREATE OR ALTER VIEW semantic.Metric_TicketResult_MinResult AS
-- TODO: Implement metric aggregation
-- Inputs: TicketResult.MinResult
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: FreeTemplate TotalState
-- Purpose: Track TotalState
-- Logic: SUM(State)
CREATE OR ALTER VIEW semantic.Metric_FreeTemplate_TotalState AS
-- TODO: Implement metric aggregation
-- Inputs: FreeTemplate.TotalState
-- Constraints: State = 1
SELECT 1 AS PlaceholderMetric
GO

-- Metric: FreeTemplate TotalPrice
-- Purpose: Track TotalPrice
-- Logic: SUM(Price)
CREATE OR ALTER VIEW semantic.Metric_FreeTemplate_TotalPrice AS
-- TODO: Implement metric aggregation
-- Inputs: FreeTemplate.TotalPrice
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: FreeTemplate TotalSalesTypeID
-- Purpose: Track TotalSalesTypeID
-- Logic: SUM(SalesTypeID)
CREATE OR ALTER VIEW semantic.Metric_FreeTemplate_TotalSalesTypeID AS
-- TODO: Implement metric aggregation
-- Inputs: FreeTemplate.TotalSalesTypeID
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: FreeTemplateAdvertisement RecordCount
-- Purpose: Track RecordCount
-- Logic: COUNT(*)
CREATE OR ALTER VIEW semantic.Metric_FreeTemplateAdvertisement_RecordCount AS
-- TODO: Implement metric aggregation
-- Inputs: FreeTemplateAdvertisement.RecordCount
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: FreeTemplateAdvertisement DistinctProductCount
-- Purpose: Track DistinctProductCount
-- Logic: COUNT(DISTINCT ProductID)
CREATE OR ALTER VIEW semantic.Metric_FreeTemplateAdvertisement_DistinctProductCount AS
-- TODO: Implement metric aggregation
-- Inputs: FreeTemplateAdvertisement.DistinctProductCount
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: FreeTemplateAdvertisement DistinctVersionSectionCount
-- Purpose: Track DistinctVersionSectionCount
-- Logic: COUNT(DISTINCT VersionSectionID)
CREATE OR ALTER VIEW semantic.Metric_FreeTemplateAdvertisement_DistinctVersionSectionCount AS
-- TODO: Implement metric aggregation
-- Inputs: FreeTemplateAdvertisement.DistinctVersionSectionCount
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: BatchActionHistoryAnalysis RecordCount
-- Purpose: Track RecordCount
-- Logic: COUNT(*)
CREATE OR ALTER VIEW semantic.Metric_BatchActionHistoryAnalysis_RecordCount AS
-- TODO: Implement metric aggregation
-- Inputs: BatchActionHistoryAnalysis.RecordCount
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: BatchActionHistoryAnalysis DistinctEntityCount
-- Purpose: Track DistinctEntityCount
-- Logic: COUNT(DISTINCT EntityID)
CREATE OR ALTER VIEW semantic.Metric_BatchActionHistoryAnalysis_DistinctEntityCount AS
-- TODO: Implement metric aggregation
-- Inputs: BatchActionHistoryAnalysis.DistinctEntityCount
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: BatchActionHistoryAnalysis DistinctEntityTypeCount
-- Purpose: Track DistinctEntityTypeCount
-- Logic: COUNT(DISTINCT EntityType)
CREATE OR ALTER VIEW semantic.Metric_BatchActionHistoryAnalysis_DistinctEntityTypeCount AS
-- TODO: Implement metric aggregation
-- Inputs: BatchActionHistoryAnalysis.DistinctEntityTypeCount
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: TaskTarget RecordCount
-- Purpose: Track RecordCount
-- Logic: COUNT(*)
CREATE OR ALTER VIEW semantic.Metric_TaskTarget_RecordCount AS
-- TODO: Implement metric aggregation
-- Inputs: TaskTarget.RecordCount
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: TaskTarget DistinctDMKeyCount
-- Purpose: Track DistinctDMKeyCount
-- Logic: COUNT(DISTINCT DMKey)
CREATE OR ALTER VIEW semantic.Metric_TaskTarget_DistinctDMKeyCount AS
-- TODO: Implement metric aggregation
-- Inputs: TaskTarget.DistinctDMKeyCount
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: TaskTarget TotalDMKey
-- Purpose: Track TotalDMKey
-- Logic: SUM(DMKey)
CREATE OR ALTER VIEW semantic.Metric_TaskTarget_TotalDMKey AS
-- TODO: Implement metric aggregation
-- Inputs: TaskTarget.TotalDMKey
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: BatchActionHistory ActionCount
-- Purpose: Track ActionCount
-- Logic: COUNT(*)
CREATE OR ALTER VIEW semantic.Metric_BatchActionHistory_ActionCount AS
-- TODO: Implement metric aggregation
-- Inputs: BatchActionHistory.ActionCount
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: BatchActionHistory TotalRowsAffected
-- Purpose: Track TotalRowsAffected
-- Logic: SUM(RowsAffected)
CREATE OR ALTER VIEW semantic.Metric_BatchActionHistory_TotalRowsAffected AS
-- TODO: Implement metric aggregation
-- Inputs: BatchActionHistory.TotalRowsAffected
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: BatchActionHistory AverageRowsAffected
-- Purpose: Track AverageRowsAffected
-- Logic: AVG(RowsAffected)
CREATE OR ALTER VIEW semantic.Metric_BatchActionHistory_AverageRowsAffected AS
-- TODO: Implement metric aggregation
-- Inputs: BatchActionHistory.AverageRowsAffected
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: TicketCommunication CommunicationCount
-- Purpose: Track CommunicationCount
-- Logic: COUNT(*)
CREATE OR ALTER VIEW semantic.Metric_TicketCommunication_CommunicationCount AS
-- TODO: Implement metric aggregation
-- Inputs: TicketCommunication.CommunicationCount
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: TicketCommunication DistinctTicketCount
-- Purpose: Track DistinctTicketCount
-- Logic: COUNT(DISTINCT TicketID)
CREATE OR ALTER VIEW semantic.Metric_TicketCommunication_DistinctTicketCount AS
-- TODO: Implement metric aggregation
-- Inputs: TicketCommunication.DistinctTicketCount
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: TicketCommunication CommunicationsPerTicket
-- Purpose: Track CommunicationsPerTicket
-- Logic: COUNT(*) * 1.0 / NULLIF(COUNT(DISTINCT TicketID), 0)
CREATE OR ALTER VIEW semantic.Metric_TicketCommunication_CommunicationsPerTicket AS
-- TODO: Implement metric aggregation
-- Inputs: TicketCommunication.CommunicationsPerTicket
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: KeywordMapping TotalRecords
-- Purpose: Track TotalRecords
-- Logic: COUNT(*)
CREATE OR ALTER VIEW semantic.Metric_KeywordMapping_TotalRecords AS
-- TODO: Implement metric aggregation
-- Inputs: KeywordMapping.TotalRecords
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: KeywordMapping DistinctActivityCount
-- Purpose: Track DistinctActivityCount
-- Logic: COUNT(DISTINCT ActivityID)
CREATE OR ALTER VIEW semantic.Metric_KeywordMapping_DistinctActivityCount AS
-- TODO: Implement metric aggregation
-- Inputs: KeywordMapping.DistinctActivityCount
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: KeywordMapping DistinctKeywordCount
-- Purpose: Track DistinctKeywordCount
-- Logic: COUNT(DISTINCT KeywordID)
CREATE OR ALTER VIEW semantic.Metric_KeywordMapping_DistinctKeywordCount AS
-- TODO: Implement metric aggregation
-- Inputs: KeywordMapping.DistinctKeywordCount
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: Step1 RecordCount
-- Purpose: Track RecordCount
-- Logic: COUNT(*)
CREATE OR ALTER VIEW semantic.Metric_Step1_RecordCount AS
-- TODO: Implement metric aggregation
-- Inputs: Step1.RecordCount
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: Step1 SumActivityID
-- Purpose: Track SumActivityID
-- Logic: SUM(ActivityID)
CREATE OR ALTER VIEW semantic.Metric_Step1_SumActivityID AS
-- TODO: Implement metric aggregation
-- Inputs: Step1.SumActivityID
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO

-- Metric: Step1 AverageActivityID
-- Purpose: Track AverageActivityID
-- Logic: AVG(ActivityID)
CREATE OR ALTER VIEW semantic.Metric_Step1_AverageActivityID AS
-- TODO: Implement metric aggregation
-- Inputs: Step1.AverageActivityID
-- Constraints: 
SELECT 1 AS PlaceholderMetric
GO
