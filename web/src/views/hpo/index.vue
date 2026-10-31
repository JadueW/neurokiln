<template>
  <div class="app-container">
    <h2 class="page-title">{{ $t('hpo.title') }}</h2>
    <p class="page-desc">{{ $t('hpo.description') }}</p>

    <el-card shadow="never" style="margin-bottom:16px">
      <el-form :inline="true" label-width="92px">
        <el-form-item :label="$t('hpo.dataset')">
          <el-select :model-value="selection.datasetId" style="width:240px" @change="onDatasetChange">
            <el-option v-for="item in datasets" :key="item.id" :label="tr(item.name)" :value="item.id" :disabled="!item.ready" />
          </el-select>
        </el-form-item>
        <el-form-item :label="$t('hpo.subjects')">
          <el-select v-model="selection.subjectIds" multiple collapse-tags clearable style="width:170px" :placeholder="$t('hpo.defaultSubjects')">
            <el-option v-for="subject in subjectOptions" :key="subject" :label="subject" :value="subject" />
          </el-select>
        </el-form-item>
        <el-form-item :label="$t('hpo.protocol')">
          <el-select v-model="form.protocol" style="width:180px">
            <el-option :label="$t('hpo.protocolWithinSubject')" value="within_subject" />
            <el-option :label="$t('hpo.protocolLoso')" value="loso" />
            <el-option :label="$t('hpo.protocolCrossSession')" value="cross_session" />
          </el-select>
        </el-form-item>
        <el-form-item :label="$t('hpo.folds')" v-if="form.protocol === 'within_subject'"><el-input-number v-model="form.folds" :min="2" :max="10" /></el-form-item>
        <el-form-item :label="$t('hpo.repeats')" v-if="form.protocol === 'within_subject'"><el-input-number v-model="form.repeats" :min="1" :max="5" /></el-form-item>
        <el-form-item :label="$t('hpo.seeds')" v-if="form.protocol === 'within_subject'"><el-input v-model="seedText" style="width:120px" /></el-form-item>
        <el-form-item :label="$t('hpo.classCount')" v-if="classOptions.length">
          <el-select v-model="form.classes" style="width:100px">
            <el-option v-for="count in classOptions" :key="count" :label="count" :value="count" />
          </el-select>
        </el-form-item>
        <el-form-item :label="$t('hpo.trialsPerClass')" v-if="isSynthetic"><el-input-number v-model="form.trialsPerClass" :min="10" :max="200" :step="10" /></el-form-item>
        <el-form-item :label="$t('hpo.template')">
          <el-select v-model="form.templateId" style="width:220px">
            <el-option
              v-for="item in templates"
              :key="item.id"
              :label="item.name"
              :value="item.id"
              :disabled="item.valid === false"
            >
              <span>{{ tr(item.name) }}</span>
              <el-tag v-if="item.family === 'custom'" size="small" type="success" style="float:right;margin-left:12px">{{ $t('hpo.custom') }}</el-tag>
              <el-tag v-if="item.deep" size="small" type="warning" style="float:right;margin-left:12px">{{ $t('hpo.deep') }}</el-tag>
              <el-tag
                v-if="item.valid === false"
                size="small"
                type="danger"
                style="float:right;margin-left:12px"
                :title="item.error"
              >{{ $t('hpo.unavailable') }}</el-tag>
            </el-option>
          </el-select>
        </el-form-item>
        <el-form-item :label="$t('hpo.objectives')">
          <el-checkbox-group v-model="form.objectives">
            <el-checkbox
              v-for="item in objectivesCatalog"
              :key="item.name"
              :value="item.name"
              :disabled="item.name === 'balanced_accuracy'"
            >
              {{ tr(item.label) }}{{ item.direction === 'min' ? ' ↓' : ' ↑' }}
            </el-checkbox>
          </el-checkbox-group>
        </el-form-item>
        <el-form-item :label="$t('hpo.budget')">
          <el-tooltip :content="$t('hpo.budgetHint')" placement="top">
            <el-input-number v-model="form.budget" :min="4" :max="120" :step="4" />
          </el-tooltip>
        </el-form-item>
        <el-form-item :label="$t('hpo.population')">
          <el-input-number v-model="form.population" :min="3" :max="maxPopulation" />
        </el-form-item>
        <el-form-item :label="$t('hpo.proxyFolds')">
          <el-tooltip :content="$t('hpo.proxyFoldsHint')" placement="top">
            <el-input-number v-model="form.proxyFolds" :min="1" :max="10" />
          </el-tooltip>
        </el-form-item>
        <el-form-item :label="$t('hpo.fullReevaluation')">
          <el-tooltip :content="$t('hpo.fullCandidatesHint')" placement="top">
            <el-input-number v-model="form.fullCandidates" :min="1" :max="maxFullCandidates" />
          </el-tooltip>
        </el-form-item>
        <el-form-item :label="$t('hpo.searchSeed')"><el-input-number v-model="form.seed" :min="0" :max="999999" /></el-form-item>
        <el-form-item :label="$t('hpo.experimentName')">
          <el-input v-model="form.name" :placeholder="$t('hpo.namePlaceholder')" style="width:200px" />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" icon="Aim" :loading="submitting" :disabled="jobActive" @click="run">{{ $t('hpo.startSearch') }}</el-button>
          <el-button icon="Timer" @click="openJobs">{{ $t('hpo.jobRecords') }}</el-button>
        </el-form-item>
      </el-form>
      <el-alert v-if="currentTemplate" type="info" :closable="false" style="margin-bottom:8px" :title="tr(currentTemplate.description)" />
      <el-alert
        v-if="currentTemplate?.family === 'custom'"
        type="success"
        :closable="false"
        style="margin-bottom:8px"
        :title="$t('hpo.customTemplateNotice', { id: currentTemplate.custom_pipeline_id })"
      />
      <el-alert
        v-if="currentTemplate?.valid === false"
        type="error"
        :closable="false"
        style="margin-bottom:8px"
        :title="$t('hpo.templateUnavailable', { error: currentTemplate.error })"
      />
      <el-alert v-if="currentTemplate?.deep" type="warning" :closable="false"
        :title="$t('hpo.deepWarning')" />
    </el-card>

    <el-row v-if="currentTemplate" :gutter="16" style="margin-bottom:16px">
      <el-col :xs="24" :lg="14" class="hpo-col">
        <el-card shadow="never">
          <template #header>
            <div class="section-head">
              <b>{{ $t('hpo.searchSpace') }}</b>
              <span class="text-muted">{{ currentTemplate.name }} · {{ $t('hpo.dims', { count: currentTemplate.space.length }) }}</span>
            </div>
          </template>
          <el-table :data="currentTemplate.space" size="small" border>
            <el-table-column :label="$t('hpo.param')" min-width="150">
              <template #default="{ row }">
                <b>{{ tr(row.label) }}</b>
                <span class="mono text-muted" style="font-size:11px;margin-left:6px">{{ row.name }}</span>
              </template>
            </el-table-column>
            <el-table-column :label="$t('hpo.type')" width="80" align="center">
              <template #default="{ row }">{{ specKindLabel(row.kind) }}</template>
            </el-table-column>
            <el-table-column :label="$t('hpo.rangeOptions')" min-width="180">
              <template #default="{ row }">{{ specRange(row) }}</template>
            </el-table-column>
            <el-table-column :label="$t('hpo.default')" width="100" align="center">
              <template #default="{ row }"><span class="mono">{{ row.default }}</span></template>
            </el-table-column>
          </el-table>
        </el-card>
      </el-col>
      <el-col :xs="24" :lg="10" class="hpo-col">
        <el-card shadow="never">
          <template #header>
            <div class="section-head">
              <b>{{ $t('hpo.configPreview') }}</b>
              <el-button size="small" icon="Download" @click="exportConfig">{{ $t('hpo.exportConfig') }}</el-button>
            </div>
          </template>
          <pre class="json-preview" style="max-height:300px">{{ configJson }}</pre>
        </el-card>
      </el-col>
    </el-row>

    <el-card v-if="job" shadow="never" style="margin-bottom:16px">
      <div class="job-row">
        <div class="job-title">
          <b>{{ $t('hpo.backgroundJob') }}</b>
          <span class="mono text-muted">{{ job.id.slice(0, 8) }}</span>
          <el-tag size="small" :type="jobStatusType">{{ jobStatusLabel }}</el-tag>
          <span class="text-muted job-name">{{ jobName(job) }}</span>
        </div>
        <div class="job-actions">
          <el-button v-if="jobActive" size="small" type="danger" plain icon="Close" @click="cancelActiveJob">{{ $t('hpo.cancelJob') }}</el-button>
          <el-button v-if="job.status === 'completed' && job.result" size="small" type="primary" @click="showActiveJob">{{ $t('hpo.viewResult') }}</el-button>
          <el-button v-if="!jobActive" size="small" @click="job = null">{{ $t('hpo.collapse') }}</el-button>
        </div>
      </div>
      <el-progress
        :percentage="jobProgress"
        :status="jobProgressStatus"
        :stroke-width="10"
        style="margin-top:10px"
      />
      <p class="job-message">{{ tr(job.message) || $t('hpo.waitingSchedule') }}</p>
      <el-alert
        v-if="job.status === 'failed'"
        type="error"
        :closable="false"
        :title="jobErrorMessage(job, $t('hpo.jobFailed'))"
      />
      <el-alert
        v-if="job.status === 'cancelled'"
        type="info"
        :closable="false"
        :title="$t('hpo.jobCancelledNotice')"
      />
    </el-card>

    <el-alert v-if="promoted" type="success" :closable="false" style="margin-bottom:16px">
      <template #title>
        {{ $t('hpo.savedAsPipeline') }} <b>{{ promoted.name }}</b>
        <span class="mono text-muted" style="margin-left:6px">custom:{{ promoted.id }}</span>
      </template>
      <div class="result-actions">
        <el-button size="small" type="primary" link @click="goArena">{{ $t('hpo.reviewInArena') }}</el-button>
        <el-button size="small" type="primary" link @click="goPipeline">{{ $t('hpo.goPipeline') }}</el-button>
      </div>
    </el-alert>

    <template v-if="result">
      <el-alert type="success" :closable="false" style="margin-bottom:16px">
        <template #title>
          {{ tr(result.dataset.name) }} · {{ protocolLabel(result.protocol.name) }} ·
          {{ $t('hpo.budgetSummary', { budget: result.search.budget, actual: result.search.evaluations }) }} ·
          {{ formatDuration(result.duration_ms) }}
          <template v-if="activeJobId"> · {{ $t('hpo.jobLabel') }} <span class="mono">{{ activeJobId.slice(0, 8) }}</span></template>
        </template>
        <div class="result-actions">
          <el-button size="small" type="primary" link icon="Star" :disabled="!result.best" @click="openSave">{{ $t('hpo.saveAsPipeline') }}</el-button>
          <el-button size="small" type="primary" link icon="Download" @click="exportResult">{{ $t('hpo.exportResultJson') }}</el-button>
        </div>
      </el-alert>

      <el-row :gutter="16">
        <el-col :xs="24" :lg="10" class="hpo-col">
          <el-card shadow="never">
            <template #header>
              <div class="section-head">
                <b>{{ $t('hpo.bestCandidate') }}</b>
                <span v-if="result.best" class="text-muted">
                  #{{ result.best.index }} · {{ stageLabel(result.best.stage) }}
                </span>
              </div>
            </template>
            <template v-if="result.best">
              <div class="metric-grid">
                <div class="metric">
                  <div class="metric-value">{{ formatMetric(result.best.metrics.balanced_accuracy) }}</div>
                  <div class="metric-label">Balanced Accuracy</div>
                </div>
                <div class="metric">
                  <div class="metric-value">{{ formatMetric(result.best.metrics.cohen_kappa) }}</div>
                  <div class="metric-label">Cohen Kappa</div>
                </div>
                <div class="metric">
                  <div class="metric-value">{{ formatMetric(result.best.metrics.channel_count, 0) }}</div>
                  <div class="metric-label">{{ $t('hpo.channelCount') }}</div>
                </div>
                <div class="metric">
                  <div class="metric-value">{{ formatDuration(result.best.metrics.duration_ms) }}</div>
                  <div class="metric-label">{{ $t('hpo.avgFoldDuration') }}</div>
                </div>
              </div>
              <div class="objective-tags">
                <el-tag v-for="item in result.objectives" :key="item.name" size="small" :type="item.direction === 'max' ? 'success' : 'warning'">
                  {{ tr(item.label) }} {{ item.direction === 'max' ? '↑' : '↓' }}
                </el-tag>
                <span class="text-muted" style="font-size:12px">{{ $t('hpo.baAlwaysIncluded') }}</span>
              </div>
              <el-table :data="bestParamRows" size="small" border style="margin-top:12px">
                <el-table-column :label="$t('hpo.param')" min-width="150">
                  <template #default="{ row }">
                    <b>{{ tr(row.label) }}</b>
                    <span class="mono text-muted" style="font-size:11px;margin-left:6px">{{ row.name }}</span>
                  </template>
                </el-table-column>
                <el-table-column :label="$t('hpo.value')" min-width="120">
                  <template #default="{ row }"><span class="mono">{{ row.value }}</span></template>
                </el-table-column>
              </el-table>
              <el-collapse v-if="result.best.graph" style="margin-top:12px">
                <el-collapse-item :title="$t('hpo.recommendedGraph')" name="graph">
                  <el-table :data="result.best.graph.nodes" size="small" border>
                    <el-table-column prop="id" :label="$t('hpo.node')" width="90" />
                    <el-table-column prop="type" :label="$t('hpo.type')" min-width="180" />
                    <el-table-column :label="$t('hpo.param')" min-width="180">
                      <template #default="{ row }">
                        <span class="mono" style="font-size:11px">{{ paramsSummary(row.params) }}</span>
                      </template>
                    </el-table-column>
                  </el-table>
                </el-collapse-item>
              </el-collapse>
            </template>
            <el-empty v-else :description="$t('hpo.noSuccessfulCandidate')" />
          </el-card>
        </el-col>
        <el-col :xs="24" :lg="14" class="hpo-col">
          <el-card shadow="never">
            <template #header>
              <div class="section-head">
                <b>{{ $t('hpo.paretoFront') }}</b>
                <el-select v-model="scatterX" size="small" style="width:180px">
                  <el-option :label="$t('hpo.axisChannels')" value="channel_count" />
                  <el-option :label="$t('hpo.axisKappa')" value="cohen_kappa" />
                  <el-option :label="$t('hpo.axisDuration')" value="duration_ms" />
                </el-select>
              </div>
            </template>
            <div ref="chartEl" style="height:360px;width:100%" />
            <p class="chart-note">{{ $t('hpo.chartNote') }}</p>
          </el-card>
        </el-col>
      </el-row>

      <el-card shadow="never" style="margin-bottom:16px">
        <template #header>
          <div class="section-head">
            <b>{{ $t('hpo.paretoCandidates') }}</b>
            <span class="text-muted">{{ $t('hpo.nonDominated', { count: result.pareto.length }) }}</span>
          </div>
        </template>
        <el-table :data="result.pareto" border stripe size="small">
          <el-table-column prop="index" label="#" width="60" align="center" />
          <el-table-column :label="$t('hpo.stage')" width="90" align="center">
            <template #default="{ row }">
              <el-tag size="small" :type="row.stage === 'full' ? 'success' : 'info'">{{ stageLabel(row.stage) }}</el-tag>
            </template>
          </el-table-column>
          <el-table-column label="Balanced Accuracy" width="150" align="center">
            <template #default="{ row }"><b>{{ formatMetric(row.metrics.balanced_accuracy) }}</b></template>
          </el-table-column>
          <el-table-column label="Kappa" width="100" align="center">
            <template #default="{ row }">{{ formatMetric(row.metrics.cohen_kappa) }}</template>
          </el-table-column>
          <el-table-column :label="$t('hpo.channelCount')" width="90" align="center">
            <template #default="{ row }">{{ formatMetric(row.metrics.channel_count, 0) }}</template>
          </el-table-column>
          <el-table-column :label="$t('hpo.foldDuration')" width="110" align="center">
            <template #default="{ row }">{{ formatDuration(row.metrics.duration_ms) }}</template>
          </el-table-column>
          <el-table-column :label="$t('hpo.param')" min-width="320">
            <template #default="{ row }"><span class="mono" style="font-size:11px">{{ paramsSummary(row.params) }}</span></template>
          </el-table-column>
          <template #empty>
            <span class="text-muted">{{ $t('hpo.noParetoCandidates') }}</span>
          </template>
        </el-table>
      </el-card>

      <el-card v-if="result.failures && result.failures.failed" shadow="never" style="margin-bottom:16px">
        <template #header>
          <div class="section-head">
            <b>{{ $t('hpo.failureSummary') }}</b>
            <span class="text-muted">
              {{ $t('hpo.failureSummaryText', { failed: result.failures.failed, evaluations: result.failures.evaluations }) }}
            </span>
          </div>
        </template>
        <el-table :data="result.failures.groups" border stripe size="small">
          <el-table-column prop="count" :label="$t('hpo.count')" width="80" align="center" />
          <el-table-column prop="reason" :label="$t('hpo.reason')" min-width="280" />
          <el-table-column :label="$t('hpo.example')" min-width="320">
            <template #default="{ row }">
              <span class="mono" style="font-size:11px">{{ row.example }}</span>
            </template>
          </el-table-column>
        </el-table>
      </el-card>

      <el-card shadow="never" style="margin-bottom:16px">
        <template #header>
          <div class="section-head">
            <b>{{ $t('hpo.searchAudit') }}</b>
            <div class="audit-tools">
              <span class="text-muted">{{ $t('hpo.records', { count: result.history.length }) }}</span>
              <el-select v-model="historyStage" size="small" style="width:120px">
                <el-option :label="$t('hpo.allStages')" value="all" />
                <el-option :label="$t('hpo.proxyFilter')" value="proxy" />
                <el-option :label="$t('hpo.fullReevaluation')" value="full" />
              </el-select>
            </div>
          </div>
        </template>
        <el-table :data="historyRows" border stripe size="small" max-height="480" row-key="index">
          <el-table-column type="expand">
            <template #default="{ row }">
              <div class="expand-box">
                <div v-if="row.error" class="error-line">{{ $t('hpo.failureReason', { error: row.error }) }}</div>
                <div class="expand-title">{{ $t('hpo.param') }}</div>
                <pre class="json-preview" style="max-height:200px">{{ formatJson(row.params) }}</pre>
                <div class="expand-title">{{ $t('hpo.normalizedVector') }}</div>
                <div class="mono vector-line">[{{ row.vector.map((item) => item.toFixed(3)).join(', ') }}]</div>
                <div class="expand-title">{{ $t('hpo.splitSignatures', { count: row.split_signatures.length }) }}</div>
                <div class="mono vector-line">{{ row.split_signatures.join(' · ') }}</div>
              </div>
            </template>
          </el-table-column>
          <el-table-column prop="index" label="#" width="55" align="center" />
          <el-table-column :label="$t('hpo.stage')" width="80" align="center">
            <template #default="{ row }">
              <el-tag size="small" :type="row.stage === 'full' ? 'success' : 'info'">{{ stageLabel(row.stage) }}</el-tag>
            </template>
          </el-table-column>
          <el-table-column prop="generation" :label="$t('hpo.generation')" width="55" align="center" />
          <el-table-column :label="$t('hpo.role')" width="90" align="center">
            <template #default="{ row }">{{ roleLabel(row.role) }}</template>
          </el-table-column>
          <el-table-column label="BA" width="90" align="center">
            <template #default="{ row }">{{ formatMetric(row.metrics?.balanced_accuracy) }}</template>
          </el-table-column>
          <el-table-column label="Kappa" width="90" align="center">
            <template #default="{ row }">{{ formatMetric(row.metrics?.cohen_kappa) }}</template>
          </el-table-column>
          <el-table-column :label="$t('hpo.channels')" width="70" align="center">
            <template #default="{ row }">{{ formatMetric(row.metrics?.channel_count, 0) }}</template>
          </el-table-column>
          <el-table-column :label="$t('hpo.duration')" width="95" align="center">
            <template #default="{ row }">{{ formatDuration(row.duration_ms) }}</template>
          </el-table-column>
          <el-table-column :label="$t('hpo.param')" min-width="260">
            <template #default="{ row }"><span class="mono" style="font-size:11px">{{ paramsSummary(row.params) }}</span></template>
          </el-table-column>
          <el-table-column :label="$t('hpo.status')" width="80" align="center">
            <template #default="{ row }">
              <el-tag v-if="row.error" size="small" type="danger">{{ $t('hpo.statusFailed') }}</el-tag>
              <el-tag v-else-if="frontIndexes.has(row.index)" size="small" type="danger" effect="plain">Pareto</el-tag>
              <el-tag v-else size="small" type="success" effect="plain">{{ $t('hpo.statusValid') }}</el-tag>
            </template>
          </el-table-column>
        </el-table>
      </el-card>

      <el-card shadow="never">
        <template #header><b>{{ $t('hpo.searchConfig') }}</b></template>
        <el-descriptions :column="3" border size="small">
          <el-descriptions-item :label="$t('hpo.searchAlgorithm')">{{ result.search.algorithm }}</el-descriptions-item>
          <el-descriptions-item :label="$t('hpo.optimizationSeed')">{{ result.search.seed }}</el-descriptions-item>
          <el-descriptions-item :label="$t('hpo.budgetActual')">{{ result.search.budget }} / {{ result.search.evaluations }}</el-descriptions-item>
          <el-descriptions-item :label="$t('hpo.proxyEvaluations')">{{ result.search.proxy_evaluations }}</el-descriptions-item>
          <el-descriptions-item :label="$t('hpo.fullReevaluation')">{{ result.search.full_evaluations }}</el-descriptions-item>
          <el-descriptions-item :label="$t('hpo.failedEvaluations')">{{ result.search.failed_evaluations ?? '-' }}</el-descriptions-item>
          <el-descriptions-item :label="$t('hpo.populationGenerations')">{{ result.search.population }} / {{ result.search.generations }}</el-descriptions-item>
          <el-descriptions-item :label="$t('hpo.proxyFoldsLabel')">{{ result.search.proxy_folds }}{{ $t('hpo.proxyIndices', { indices: result.search.proxy_indices.join(', ') }) }}</el-descriptions-item>
          <el-descriptions-item :label="$t('hpo.fullSplits')">{{ result.search.full_splits }}</el-descriptions-item>
          <el-descriptions-item :label="$t('hpo.paretoReevalLimit')">{{ result.search.full_candidates }}</el-descriptions-item>
          <el-descriptions-item :label="$t('hpo.datasetSignature')">
            <span class="mono">{{ String(result.dataset.signature || '-').slice(0, 16) }}</span>
          </el-descriptions-item>
          <el-descriptions-item :label="$t('hpo.trialsChannelsClasses')">
            {{ result.dataset.n_trials }} / {{ result.dataset.n_channels }} / {{ result.dataset.n_classes }}
          </el-descriptions-item>
          <el-descriptions-item :label="$t('hpo.protocol')">
            {{ protocolLabel(result.protocol.name) }} · {{ $t('hpo.protocolSummary', { folds: result.protocol.folds, repeats: result.protocol.repeats, seeds: result.protocol.seeds.join(', ') }) }}
          </el-descriptions-item>
        </el-descriptions>
        <el-alert style="margin-top:16px" type="info" :closable="false" :title="tr(result.note)" />
      </el-card>
    </template>

    <el-dialog v-model="saveVisible" :title="$t('hpo.saveAsPipeline')" width="520px">
      <el-form label-width="70px">
        <el-form-item :label="$t('hpo.name')"><el-input v-model="saveForm.name" maxlength="80" show-word-limit /></el-form-item>
        <el-form-item :label="$t('hpo.descriptionField')">
          <el-input v-model="saveForm.description" type="textarea" :rows="3" maxlength="500" show-word-limit />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="saveVisible = false">{{ $t('common.cancel') }}</el-button>
        <el-button type="primary" :loading="promoteLoading" @click="confirmSave">{{ $t('common.save') }}</el-button>
      </template>
    </el-dialog>

    <el-drawer v-model="jobsVisible" :title="$t('hpo.jobRecordsTitle')" size="min(860px, 94%)">
      <div class="filter-bar">
        <el-button icon="Refresh" :loading="jobsLoading" @click="loadJobs">{{ $t('common.refresh') }}</el-button>
        <span class="text-muted" style="font-size:12px">{{ $t('hpo.jobNote') }}</span>
      </div>
      <el-table :data="hpoJobs" border stripe size="small" v-loading="jobsLoading">
        <el-table-column :label="$t('hpo.jobLabel')" width="100">
          <template #default="{ row }"><span class="mono">{{ row.id.slice(0, 8) }}</span></template>
        </el-table-column>
        <el-table-column :label="$t('hpo.status')" width="90" align="center">
          <template #default="{ row }">
            <el-tag size="small" :type="statusMeta(row.status).type">{{ statusMeta(row.status).label }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column :label="$t('hpo.experimentName')" min-width="160">
          <template #default="{ row }">{{ jobName(row) }}</template>
        </el-table-column>
        <el-table-column :label="$t('hpo.progress')" width="130">
          <template #default="{ row }">
            <el-progress :percentage="Math.round((row.progress || 0) * 100)" :stroke-width="8" />
          </template>
        </el-table-column>
        <el-table-column :label="$t('hpo.info')" min-width="180">
          <template #default="{ row }">
            <div>{{ tr(row.message) || '-' }}</div>
            <div v-if="row.error" class="text-muted" style="font-size:11px">{{ row.error }}</div>
          </template>
        </el-table-column>
        <el-table-column :label="$t('hpo.submittedAt')" width="170" align="center">
          <template #default="{ row }">{{ formatTime(row.created_at) }}</template>
        </el-table-column>
        <el-table-column :label="$t('hpo.actions')" width="130" align="center" fixed="right">
          <template #default="{ row }">
            <el-button v-if="row.status === 'completed'" link type="primary" @click="viewJob(row)">{{ $t('hpo.view') }}</el-button>
            <el-button v-if="['queued', 'running'].includes(row.status)" link type="danger" @click="cancelJobRow(row)">{{ $t('common.cancel') }}</el-button>
          </template>
        </el-table-column>
        <template #empty>
          <el-empty :description="$t('hpo.noJobs')" />
        </template>
      </el-table>
    </el-drawer>
  </div>
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { tr } from '@/i18n/catalog'
import {
  cancelJob,
  getDatasets,
  getHPOTemplates,
  getJob,
  getJobs,
  promoteHPOResult,
  submitHPOJob,
} from '@/api'
import { createChart } from '@/utils/chart'
import { downloadJson, formatDuration, formatMetric, formatTime } from '@/utils/format'
import { jobErrorMessage } from '@/utils/errors'
import { selection, setDataset, subjectChoices } from '@/utils/selection'

const JOB_STATUS = {
  queued: { key: 'hpo.statusQueued', type: 'info' },
  running: { key: 'hpo.statusRunning', type: 'warning' },
  completed: { key: 'hpo.statusCompleted', type: 'success' },
  failed: { key: 'hpo.statusFailed', type: 'danger' },
  cancelled: { key: 'hpo.statusCancelled', type: 'info' },
}

const { t } = useI18n()
const router = useRouter()
const route = useRoute()
const datasets = ref([])
const templates = ref([])
const objectivesCatalog = ref([])
const result = ref(null)
const submitting = ref(false)
const seedText = ref('0')
const chartEl = ref(null)
const scatterX = ref('channel_count')
const historyStage = ref('all')
const job = ref(null)
const activeJobId = ref('')
const jobsVisible = ref(false)
const jobsList = ref([])
const jobsLoading = ref(false)
const saveVisible = ref(false)
const promoteLoading = ref(false)
const promoted = ref(null)
const saveForm = reactive({ name: '', description: '' })
let chartCtl = null
let pollTimer = null
let pollToken = 0

const form = reactive({
  protocol: 'within_subject', folds: 5, repeats: 1, classes: 2,
  trialsPerClass: 40, templateId: 'csp_lda',
  objectives: ['balanced_accuracy', 'channel_count'],
  budget: 16, population: 6, proxyFolds: 2, fullCandidates: 3, seed: 2026, name: '',
})

const current = computed(() => datasets.value.find((item) => item.id === selection.datasetId))
const currentTemplate = computed(() => templates.value.find((item) => item.id === form.templateId))
const subjectOptions = computed(() => subjectChoices(current.value))
const classOptions = computed(() => {
  const count = current.value?.classes || 2
  return count > 2 ? Array.from({ length: count - 1 }, (_, index) => index + 2) : []
})
const isSynthetic = computed(() => current.value?.adapter === 'builtin')
const maxPopulation = computed(() => Math.max(3, Math.min(16, form.budget - 1)))
const maxFullCandidates = computed(() => Math.max(1, Math.min(10, form.budget - 1)))
const jobActive = computed(() => !!job.value && ['queued', 'running'].includes(job.value.status))
const jobProgress = computed(() => Math.round((job.value?.progress || 0) * 100))
const jobStatusType = computed(() => statusMeta(job.value?.status).type)
const jobStatusLabel = computed(() => statusMeta(job.value?.status).label)
const jobProgressStatus = computed(() => {
  if (job.value?.status === 'completed') return 'success'
  if (job.value?.status === 'failed') return 'exception'
  if (job.value?.status === 'cancelled') return 'warning'
  return ''
})
const frontIndexes = computed(() => new Set((result.value?.pareto || []).map((item) => item.index)))
const resultSpaceLabels = computed(() => Object.fromEntries(
  (result.value?.template?.space || []).map((item) => [item.name, tr(item.label)]),
))
const bestParamRows = computed(() => {
  const best = result.value?.best
  if (!best) return []
  const space = result.value?.template?.space || []
  return Object.entries(best.params).map(([name, value]) => ({
    name,
    label: tr(space.find((item) => item.name === name)?.label) || name,
    value,
  }))
})
const historyRows = computed(() => {
  const rows = result.value?.history || []
  return historyStage.value === 'all' ? rows : rows.filter((row) => row.stage === historyStage.value)
})
const hpoJobs = computed(() => jobsList.value.filter((item) => item.kind === 'hpo'))
const configJson = computed(() => JSON.stringify(buildPayload(), null, 2))

watch(() => form.budget, (value) => {
  if (form.population > Math.max(3, Math.min(16, value - 1))) {
    form.population = Math.max(3, Math.min(16, value - 1))
  }
  if (form.fullCandidates > Math.max(1, Math.min(10, value - 1))) {
    form.fullCandidates = Math.max(1, Math.min(10, value - 1))
  }
})

watch(scatterX, () => {
  if (result.value) renderChart()
})

function statusMeta(status) {
  return JOB_STATUS[status]
    ? { label: t(JOB_STATUS[status].key), type: JOB_STATUS[status].type }
    : { label: status || '-', type: 'info' }
}

function protocolLabel(value) {
  const labels = {
    within_subject: t('hpo.protocolWithinSubject'),
    loso: t('hpo.protocolLoso'),
    cross_session: t('hpo.protocolCrossSession'),
  }
  return labels[value] || value || '-'
}

function stageLabel(value) {
  return value === 'full' ? t('hpo.fullReevaluation') : t('hpo.proxyFilter')
}

function roleLabel(value) {
  const labels = {
    init: t('hpo.roleInit'),
    gwo: t('hpo.roleGwo'),
    local: t('hpo.roleLocal'),
    random_fill: t('hpo.roleRandomFill'),
    pareto: t('hpo.rolePareto'),
  }
  return labels[value] || value || '-'
}

function specKindLabel(kind) {
  return { float: t('hpo.kindFloat'), int: t('hpo.kindInt'), select: t('hpo.kindSelect') }[kind] || kind
}

function specRange(spec) {
  if (spec.kind === 'select') return spec.options.join(' / ')
  const suffix = spec.log ? t('hpo.rangeLog') : (spec.step ? t('hpo.rangeStep', { step: spec.step }) : '')
  return `${spec.low} – ${spec.high}${suffix}`
}

function paramsSummary(params) {
  if (!params || !Object.keys(params).length) return '-'
  return Object.entries(params)
    .map(([key, value]) => `${resultSpaceLabels.value[key] || key}=${value}`)
    .join(' · ')
}

function formatJson(value) {
  return JSON.stringify(value, null, 2)
}

function templateName(id) {
  return templates.value.find((item) => item.id === id)?.name || id || 'HPO'
}

function jobName(row) {
  return row?.request?.name || t('hpo.jobSuffix', { name: templateName(row?.request?.template_id) })
}

function syncDefaults() {
  const dataset = current.value
  if (!dataset) return
  form.classes = dataset.classes || 2
  const options = subjectChoices(dataset)
  const valid = selection.subjectIds.filter((id) => options.includes(id))
  selection.subjectIds = valid.length ? valid : options.slice(0, 1)
}

function onDatasetChange(datasetId) {
  setDataset(datasetId)
  result.value = null
  promoted.value = null
  activeJobId.value = ''
  disposeChart()
  syncDefaults()
}

function buildPayload() {
  const options = {}
  if (isSynthetic.value) options.n_trials_per_class = form.trialsPerClass
  const seeds = seedText.value
    .split(',')
    .map((item) => parseInt(item.trim(), 10))
    .filter((value) => !Number.isNaN(value))
  return {
    dataset_id: selection.datasetId,
    subject_ids: [...selection.subjectIds],
    classes: form.classes,
    options,
    protocol: form.protocol,
    folds: form.folds,
    repeats: form.repeats,
    seeds: seeds.length ? seeds : [0],
    template_id: form.templateId,
    objectives: [...form.objectives],
    budget: form.budget,
    population: form.population,
    proxy_folds: form.proxyFolds,
    full_candidates: form.fullCandidates,
    seed: form.seed,
    name: form.name,
  }
}

function validate() {
  if (!form.templateId) {
    ElMessage.warning(t('hpo.validationTemplate'))
    return false
  }
  if (form.budget <= form.population) {
    ElMessage.warning(t('hpo.validationBudget'))
    return false
  }
  if (form.protocol === 'loso' && selection.subjectIds.length < 2) {
    ElMessage.warning(t('hpo.validationLosoSubjects'))
    return false
  }
  if (form.protocol !== 'loso' && selection.subjectIds.length !== 1) {
    ElMessage.warning(t('hpo.validationSingleSubject'))
    return false
  }
  return true
}

function disposeChart() {
  chartCtl?.dispose()
  chartCtl = null
}

function renderChart() {
  if (!chartEl.value || !result.value) return
  if (!chartCtl) chartCtl = createChart(chartEl.value)
  const xKey = scatterX.value
  const front = frontIndexes.value
  const usable = (result.value.history || []).filter((item) => item.metrics)
  const groups = [
    { key: 'proxy', name: t('hpo.proxyFilter'), color: '#a0cfff', symbol: 'circle', size: 8, items: usable.filter((row) => row.stage === 'proxy' && !front.has(row.index)) },
    { key: 'full', name: t('hpo.fullReevaluation'), color: '#409eff', symbol: 'diamond', size: 11, items: usable.filter((row) => row.stage === 'full' && !front.has(row.index)) },
    { key: 'front', name: t('hpo.paretoFront'), color: '#f56c6c', symbol: 'circle', size: 16, items: usable.filter((row) => front.has(row.index)) },
  ]
  const xNames = {
    channel_count: t('hpo.channelCount'),
    cohen_kappa: 'Cohen Kappa',
    duration_ms: t('hpo.avgFoldDurationMs'),
  }
  chartCtl.chart.setOption({
    tooltip: {
      trigger: 'item',
      formatter: (params) => {
        const row = params.data?.raw
        if (!row) return params.name
        const metrics = row.metrics || {}
        return [
          `<b>#${row.index}</b> · ${stageLabel(row.stage)} · ${roleLabel(row.role)}`,
          `BA ${formatMetric(metrics.balanced_accuracy)} · Kappa ${formatMetric(metrics.cohen_kappa)}`,
          t('hpo.chartTooltipMetrics', { channels: formatMetric(metrics.channel_count, 0), duration: formatDuration(metrics.duration_ms) }),
          `<span style="font-size:11px">${paramsSummary(row.params)}</span>`,
        ].join('<br/>')
      },
    },
    legend: { top: 0, data: groups.map((group) => group.name) },
    grid: { left: 52, right: 26, top: 64, bottom: 30, containLabel: true },
    xAxis: {
      type: 'value', name: xNames[xKey], nameLocation: 'middle', nameGap: 28, scale: true,
      axisLabel: { color: '#909399' }, splitLine: { lineStyle: { type: 'dashed' } },
    },
    yAxis: {
      type: 'value', name: 'BA', min: 0, max: 1,
      nameLocation: 'middle', nameRotate: 90, nameGap: 38,
      nameTextStyle: { color: '#909399', fontSize: 11 },
      axisLabel: { color: '#909399' }, splitLine: { lineStyle: { type: 'dashed' } },
    },
    series: groups.map((group) => ({
      name: group.name,
      type: 'scatter',
      symbol: group.symbol,
      symbolSize: group.size,
      itemStyle: { color: group.color },
      emphasis: { focus: 'series' },
      label: group.key === 'front'
        ? { show: true, formatter: (params) => `#${params.data.raw.index}`, position: 'top', fontSize: 10, color: '#f56c6c' }
        : { show: false },
      data: group.items.map((row) => ({
        value: [
          Number(Number(row.metrics[xKey]).toFixed(4)),
          Number(Number(row.metrics.balanced_accuracy).toFixed(4)),
        ],
        raw: row,
      })),
    })),
  }, true)
}

async function applyResult(payload) {
  if (!payload) return
  result.value = payload
  await nextTick()
  renderChart()
  requestAnimationFrame(() => chartCtl?.chart.resize())
}

async function showActiveJob() {
  if (!job.value?.result) return
  activeJobId.value = job.value.id
  await applyResult(job.value.result)
}

async function run() {
  if (!validate()) return
  const payload = buildPayload()
  submitting.value = true
  try {
    result.value = null
    promoted.value = null
    activeJobId.value = ''
    disposeChart()
    const submitted = await submitHPOJob(payload)
    activeJobId.value = submitted.id
    job.value = submitted
    ElMessage.success(t('hpo.jobSubmitted'))
    schedulePoll(400)
  } finally {
    submitting.value = false
  }
}

function schedulePoll(delay = 1000) {
  clearTimeout(pollTimer)
  const token = ++pollToken
  pollTimer = setTimeout(() => pollJob(token), delay)
}

function stopPolling() {
  pollToken += 1
  clearTimeout(pollTimer)
  pollTimer = null
}

async function pollJob(token) {
  if (!job.value || token !== pollToken) return
  try {
    const latest = await getJob(job.value.id)
    if (token !== pollToken) return
    job.value = latest
    if (['queued', 'running'].includes(latest.status)) {
      schedulePoll(1000)
      return
    }
    if (latest.status === 'completed') {
      activeJobId.value = latest.id
      await applyResult(latest.result)
    } else if (latest.status === 'failed') {
      ElMessage.error(jobErrorMessage(latest, t('hpo.backgroundJobFailed')))
    }
  } catch {
    // the response interceptor already reported the error; stop this poll cycle
  }
}

async function cancelActiveJob() {
  if (!job.value) return
  try {
    await cancelJob(job.value.id)
    job.value = { ...job.value, cancel_requested: true, message: t('hpo.cancelRequestedMessage') }
    ElMessage.info(t('hpo.cancelRequested'))
    schedulePoll(600)
  } catch {
    // interceptor already reported the reason
  }
}

async function loadJobs() {
  jobsLoading.value = true
  try {
    const response = await getJobs({ limit: 50 })
    jobsList.value = response.items
  } finally {
    jobsLoading.value = false
  }
}

async function openJobs() {
  jobsVisible.value = true
  await loadJobs()
}

async function viewJob(row) {
  const latest = await getJob(row.id)
  if (latest.status === 'completed') {
    job.value = latest
    activeJobId.value = latest.id
    await applyResult(latest.result)
    jobsVisible.value = false
  } else if (['queued', 'running'].includes(latest.status)) {
    job.value = latest
    jobsVisible.value = false
    schedulePoll(600)
  } else {
    ElMessage.info(t('hpo.noJobResult'))
  }
}

async function cancelJobRow(row) {
  await cancelJob(row.id)
  ElMessage.info(t('hpo.cancelRequested'))
  await loadJobs()
}

function exportConfig() {
  downloadJson('hpo_config.json', buildPayload())
  ElMessage.success(t('hpo.configExported'))
}

function exportResult() {
  if (!result.value) return
  downloadJson(`hpo-${activeJobId.value.slice(0, 8) || 'result'}.json`, result.value)
  ElMessage.success(t('hpo.resultExported'))
}

function openSave() {
  if (!activeJobId.value) {
    ElMessage.warning(t('hpo.needCompletedJob'))
    return
  }
  const definition = result.value?.best?.definition || {}
  saveForm.name = definition.name || `HPO · ${result.value?.template?.name || t('hpo.recommendedPipeline')}`
  saveForm.description = definition.note || ''
  saveVisible.value = true
}

async function confirmSave() {
  promoteLoading.value = true
  try {
    const saved = await promoteHPOResult({
      job_id: activeJobId.value,
      name: saveForm.name.trim(),
      description: saveForm.description.trim(),
    })
    promoted.value = saved
    saveVisible.value = false
    ElMessage.success(t('hpo.savedPipeline', { name: saved.name }))
  } finally {
    promoteLoading.value = false
  }
}

function goArena() {
  if (!promoted.value) return
  router.push({ path: '/arena', query: { pipeline: `custom:${promoted.value.id}` } })
}

function goPipeline() {
  router.push('/pipeline')
}

onMounted(async () => {
  const [datasetResponse, templateResponse] = await Promise.all([
    getDatasets(),
    getHPOTemplates(),
  ])
  datasets.value = datasetResponse.items
  templates.value = templateResponse.items
  objectivesCatalog.value = templateResponse.objectives
  const requested = typeof route.query.template === 'string' ? route.query.template : ''
  if (requested) {
    const match = templates.value.find((item) => item.id === requested)
    if (match && match.valid !== false) form.templateId = match.id
    else ElMessage.warning(match?.error || t('hpo.templateFallback'))
  }
  if (!templates.value.some((item) => item.id === form.templateId && item.valid !== false)) {
    form.templateId = templates.value.find((item) => item.valid !== false)?.id || 'csp_lda'
  }
  const defaults = ['balanced_accuracy', 'channel_count'].filter(
    (name) => objectivesCatalog.value.some((item) => item.name === name),
  )
  form.objectives = defaults.length ? defaults : ['balanced_accuracy']
  if (!datasets.value.some((item) => item.id === selection.datasetId)) {
    setDataset(datasets.value[0]?.id || 'synthetic')
  }
  syncDefaults()
})

onBeforeUnmount(() => {
  stopPolling()
  disposeChart()
})
</script>

<style scoped>
.hpo-col { margin-bottom: 16px; }
.job-row { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; justify-content: space-between; }
.job-title { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.job-name { font-size: 12px; }
.job-actions { display: flex; gap: 8px; }
.job-message { margin: 8px 0 10px; color: #606266; font-size: 13px; }
.result-actions { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 6px; }
.section-head { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; justify-content: space-between; }
.audit-tools { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
.metric-grid { display: grid; grid-template-columns: repeat(4, minmax(90px, 1fr)); gap: 10px; }
.metric { background: #f5f7fa; border-radius: 8px; padding: 10px 8px; text-align: center; }
.metric-value { font-size: 18px; font-weight: 700; color: #303133; }
.metric-label { margin-top: 4px; font-size: 12px; color: #909399; }
.objective-tags { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-top: 12px; }
.chart-note { margin: 6px 0 0; font-size: 12px; color: #909399; }
.expand-box { padding: 8px 16px 14px; }
.expand-title { font-size: 12px; font-weight: 600; color: #606266; margin: 10px 0 4px; }
.error-line { color: #f56c6c; font-size: 12px; margin-bottom: 6px; }
.vector-line { font-size: 11px; color: #606266; line-height: 1.7; word-break: break-all; }
@media (max-width: 768px) {
  .metric-grid { grid-template-columns: repeat(2, minmax(90px, 1fr)); }
}
</style>
