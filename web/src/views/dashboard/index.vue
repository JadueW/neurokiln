<template>
  <div class="app-container">
    <el-card shadow="never" style="margin-bottom:16px">
      <h2 class="page-title">NeuroKiln 脑炉</h2>
      <p class="page-desc" style="margin-bottom:0">{{ $t('dashboard.description') }}</p>
    </el-card>

    <el-card v-if="showOnboarding" shadow="never" class="onboarding-card" style="margin-bottom:16px">
      <div class="onboarding-body">
        <div class="onboarding-copy">
          <el-tag size="small" type="success" effect="plain">{{ $t('dashboard.onboardingTag') }}</el-tag>
          <h3>{{ $t('dashboard.onboardingTitle') }}</h3>
          <p>{{ $t('dashboard.onboardingText') }}</p>
          <div class="onboarding-actions">
            <el-button type="primary" icon="VideoPlay" :loading="demoLoading" @click="runGoldenDemo">
              {{ $t('dashboard.runDemo') }}
            </el-button>
            <el-button link type="primary" @click="dismissOnboarding">{{ $t('dashboard.skipDemo') }}</el-button>
          </div>
        </div>
        <el-steps direction="vertical" :active="3" class="onboarding-steps">
          <el-step :title="$t('dashboard.step1')" :description="$t('dashboard.step1Desc')" />
          <el-step :title="$t('dashboard.step2')" :description="$t('dashboard.step2Desc')" />
          <el-step :title="$t('dashboard.step3')" :description="$t('dashboard.step3Desc')" />
        </el-steps>
      </div>
    </el-card>

    <div class="stat-grid" style="margin-bottom:16px">
      <el-card v-for="card in cards" :key="card.label" shadow="never">
        <div class="stat-card">
          <div class="stat-icon" :style="{ background: card.color }">
            <el-icon><component :is="card.icon" /></el-icon>
          </div>
          <div>
            <div class="stat-value">{{ card.value }}</div>
            <div class="stat-label">{{ card.label }}</div>
          </div>
        </div>
      </el-card>
    </div>

    <el-row :gutter="16">
      <el-col :xs="24" :md="24" :lg="15" style="margin-bottom:16px">
        <el-card shadow="never">
          <template #header>
            <div style="display:flex;justify-content:space-between;align-items:center">
              <b>{{ $t('dashboard.recent') }}</b>
              <el-button link type="primary" @click="$router.push('/report')">{{ $t('dashboard.allReports') }}</el-button>
            </div>
          </template>
          <el-table :data="recent" size="small" border @row-dblclick="openReport">
            <el-table-column prop="id" label="#" width="55" align="center" />
            <el-table-column :label="$t('dashboard.experiment')" min-width="190">
              <template #default="{ row }">
                <el-link type="primary" @click="openReport(row)">{{ tr(row.name) }}</el-link>
                <div class="text-muted" style="font-size:12px">{{ tr(row.dataset.name) }}</div>
              </template>
            </el-table-column>
            <el-table-column :label="$t('dashboard.best')" width="90" align="center">
              <template #default="{ row }">{{ row.best?.balanced_accuracy !== null ? formatMetric(row.best?.balanced_accuracy) : '-' }}</template>
            </el-table-column>
            <el-table-column prop="n_runs" label="Runs" width="70" align="center" />
            <el-table-column :label="$t('dashboard.time')" width="165" align="center">
              <template #default="{ row }">{{ formatTime(row.created_at) }}</template>
            </el-table-column>
            <template #empty>
              <el-empty :image-size="70" :description="$t('dashboard.emptyRecent')" />
            </template>
          </el-table>
          <el-divider />
          <el-space wrap>
            <el-button type="primary" icon="Trophy" @click="$router.push('/arena')">{{ $t('dashboard.goArena') }}</el-button>
            <el-button type="success" icon="VideoPlay" :loading="demoLoading" @click="runGoldenDemo">{{ $t('dashboard.goldenDemo') }}</el-button>
            <el-button icon="Coin" @click="$router.push('/dataset')">{{ $t('dashboard.viewDatasets') }}</el-button>
            <el-button icon="Grid" @click="$router.push('/protocol')">{{ $t('dashboard.validateSplit') }}</el-button>
            <el-button icon="MagicStick" @click="$router.push('/pipeline')">{{ $t('dashboard.designPipeline') }}</el-button>
          </el-space>
        </el-card>
      </el-col>
      <el-col :xs="24" :md="24" :lg="9">
        <el-card shadow="never" style="margin-bottom:16px">
          <template #header><b>{{ $t('dashboard.status') }}</b></template>
          <el-descriptions :column="1" border size="small">
            <el-descriptions-item :label="$t('dashboard.api')">
              <el-tag :type="health.status === 'ok' ? 'success' : 'danger'" size="small">
                {{ health.status === 'ok' ? $t('dashboard.connected') : $t('dashboard.disconnected') }}
              </el-tag>
            </el-descriptions-item>
            <el-descriptions-item :label="$t('dashboard.engineVersion')">{{ health.engine || '-' }}</el-descriptions-item>
            <el-descriptions-item :label="$t('menu.dataset')">
              {{ $t('dashboard.datasetCount', { count: stats.datasets_available ?? '-' }) }}
              <el-tag v-if="stats.moabb_installed" size="small" type="success" style="margin-left:6px">{{ $t('dashboard.moabbReady') }}</el-tag>
            </el-descriptions-item>
            <el-descriptions-item label="SQLite">
              <span class="mono" style="font-size:11px">{{ stats.storage || '-' }}</span>
            </el-descriptions-item>
            <el-descriptions-item :label="$t('dashboard.leakageGuard')">
              <el-tag :type="health.leakage_guard ? 'success' : 'warning'" size="small">
                {{ health.leakage_guard ? $t('dashboard.enabled') : $t('dashboard.disabled') }}
              </el-tag>
            </el-descriptions-item>
          </el-descriptions>
        </el-card>
        <el-card shadow="never" v-if="stats.best_run">
          <template #header><b>{{ $t('dashboard.bestRun') }}</b></template>
          <div class="best-run">
            <div class="best-value">{{ formatMetric(stats.best_run.balanced_accuracy) }}</div>
            <div class="text-muted">{{ tr(stats.best_run.pipeline_name) }}</div>
            <el-link type="primary" style="margin-top:8px" @click="$router.push({ path: '/report', query: { experiment: stats.best_run.id } })">
              {{ $t('dashboard.experimentRef', { id: stats.best_run.id, name: tr(stats.best_run.name) }) }}
            </el-link>
          </div>
        </el-card>
        <el-card shadow="never" v-else>
          <el-empty :image-size="70" :description="$t('dashboard.noBest')" />
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { tr } from '@/i18n/catalog'
import { useRouter } from 'vue-router'
import { ElLoading, ElMessage } from 'element-plus'
import { getExperimentStats, getExperiments, getHealth, runGoldenDemo as runDemoRequest } from '@/api'
import { formatMetric, formatTime } from '@/utils/format'

const { t } = useI18n()
const router = useRouter()
const health = ref({})
const stats = ref({})
const recent = ref([])
const showOnboarding = ref(false)
const demoLoading = ref(false)
const DISMISS_KEY = 'neurokiln.onboarding.dismissed'

function dismissOnboarding() {
  showOnboarding.value = false
  try {
    localStorage.setItem(DISMISS_KEY, '1')
  } catch {
    /* localStorage 不可用时只关闭本次提示 */
  }
}

async function runGoldenDemo() {
  if (demoLoading.value) return
  demoLoading.value = true
  const loading = ElLoading.service({
    lock: true,
    text: t('dashboard.demoLoading'),
    background: 'rgba(0, 0, 0, 0.35)',
  })
  try {
    const result = await runDemoRequest()
    ElMessage.success(t('dashboard.demoDone', { id: result.experiment_id }))
    dismissOnboarding()
    router.push({ path: '/report', query: { experiment: result.experiment_id } })
  } catch {
    /* 错误提示由 axios 拦截器统一处理 */
  } finally {
    loading.close()
    demoLoading.value = false
  }
}

const cards = computed(() => [
  { label: t('dashboard.experiments'), value: stats.value.experiments ?? '-', icon: 'DataAnalysis', color: '#409eff' },
  { label: t('dashboard.runs'), value: stats.value.runs ?? '-', icon: 'Histogram', color: '#67c23a' },
  { label: t('dashboard.datasets'), value: stats.value.datasets_available ?? '-', icon: 'Coin', color: '#e6a23c' },
  {
    label: t('dashboard.bestBA'),
    value: stats.value.best_run ? formatMetric(stats.value.best_run.balanced_accuracy) : '-',
    icon: 'Trophy',
    color: '#9254de',
  },
])

function openReport(row) {
  router.push({ path: '/report', query: { experiment: row.id } })
}

onMounted(async () => {
  const [healthResponse, statsResponse, listResponse] = await Promise.all([
    getHealth(),
    getExperimentStats(),
    getExperiments({ limit: 5 }),
  ])
  health.value = healthResponse
  stats.value = statsResponse
  recent.value = listResponse.items
  let dismissed = false
  try {
    dismissed = localStorage.getItem(DISMISS_KEY) === '1'
  } catch {
    dismissed = false
  }
  showOnboarding.value = !dismissed && !(statsResponse.experiments > 0)
})
</script>

<style scoped>
.best-run { text-align: center; padding: 8px 0 4px; }
.best-value { font-size: 38px; font-weight: 700; color: #409eff; line-height: 1.2; }
.onboarding-card { border: 1px solid #d9ecff; background: linear-gradient(135deg, #f4f9ff 0%, #ffffff 60%); }
.onboarding-body { display: flex; flex-wrap: wrap; gap: 24px; justify-content: space-between; align-items: center; }
.onboarding-copy { flex: 1 1 420px; }
.onboarding-copy h3 { margin: 10px 0 6px; font-size: 18px; color: #1f2d3d; }
.onboarding-copy p { margin: 0 0 14px; color: #5e6d82; line-height: 1.7; max-width: 640px; }
.onboarding-actions { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; }
.onboarding-steps { flex: 0 1 320px; height: 186px; }
</style>
