<template>
  <div class="app-container">
    <h2 class="page-title">{{ $t('plugins.title') }}</h2>
    <p class="page-desc">{{ $t('plugins.descriptionPrefix') }}<span class="mono">plugin:&lt;id&gt;/&lt;pipeline&gt;</span>{{ $t('plugins.descriptionSuffix') }}</p>

    <el-alert
      v-if="status.mode"
      :type="status.trusted ? 'warning' : 'success'"
      :closable="false"
      show-icon
      style="margin-bottom:16px"
    >
      <template #title>
        {{ status.trusted ? $t('plugins.modeTrusted') : $t('plugins.modeRestricted') }}
        <span class="text-muted" style="font-weight:400;margin-left:8px">
          {{ $t('plugins.installedLabel') }} {{ status.counts?.installed ?? 0 }} · {{ $t('plugins.enabledLabel') }} {{ status.counts?.enabled ?? 0 }} · {{ $t('plugins.errorLabel') }} {{ status.counts?.errors ?? 0 }}
        </span>
      </template>
      <div style="font-size:13px;line-height:1.7">
        <template v-if="status.trusted">
          {{ $t('plugins.trustedHint') }}
        </template>
        <template v-else>
          {{ $t('plugins.restrictedHintPrefix') }}<span class="mono">NEUROKILN_PLUGIN_TRUST=1</span>{{ $t('plugins.restrictedHintSuffix') }}
        </template>
        <div class="text-muted mono" style="font-size:12px;margin-top:4px">
          {{ $t('plugins.engineLabel') }} {{ status.engine_version || '-' }} · {{ $t('plugins.schemaLabel') }} v{{ status.schema_version ?? '-' }}
          {{ $t('plugins.supportedSchemas', { versions: (status.supported_schema_versions || []).join(' / ') }) }} · {{ $t('plugins.nodeApiLabel') }} v{{ status.node_api_version ?? '-' }}
        </div>
        <div class="text-muted mono" style="font-size:12px;margin-top:2px">{{ $t('plugins.rootDir', { path: status.root || '-' }) }}</div>
      </div>
    </el-alert>

    <el-card shadow="never" style="margin-bottom:16px">
      <div class="filter-bar">
        <el-button icon="Refresh" :loading="loading" @click="refresh">{{ $t('common.refresh') }}</el-button>
        <el-button v-if="admin" type="primary" icon="Upload" @click="openUpload">{{ $t('plugins.uploadZip') }}</el-button>
        <el-tooltip v-else :content="$t('plugins.adminRequiredTooltip')" placement="top">
          <span><el-button type="primary" icon="Upload" disabled>{{ $t('plugins.uploadZip') }}</el-button></span>
        </el-tooltip>
        <span class="text-muted" style="font-size:12px">
          {{ $t('plugins.supportedRuntimes', { runtimes: (status.supported_runtimes || []).join(' / ') || '-' }) }} ·
          {{ $t('plugins.allowedPermissions', { permissions: (status.allowed_permissions || []).join(' / ') || '-' }) }}
        </span>
      </div>

      <el-tabs v-model="tab">
        <el-tab-pane :label="$t('plugins.catalogTab')" name="catalog">
          <el-empty v-if="!catalog.length" :description="$t('plugins.catalogEmpty')" />
          <div v-else class="plugin-grid">
            <el-card v-for="item in catalog" :key="item.id" shadow="hover" class="plugin-card">
              <div class="plugin-head">
                <div>
                  <div class="plugin-name">{{ tr(item.name) }}</div>
                  <div class="mono text-muted" style="font-size:12px">{{ item.id }} · v{{ item.version }}</div>
                </div>
                <div class="plugin-badges">
                  <el-tag size="small" :type="item.runtime === 'python' ? 'warning' : 'success'" effect="plain">{{ item.runtime }}</el-tag>
                  <el-tag size="small" type="info" effect="plain">{{ $t('plugins.manifestTag', { version: item.schema_version }) }}</el-tag>
                  <el-tag size="small" :type="item.compatible ? 'success' : 'danger'" effect="plain">
                    {{ item.compatible ? $t('plugins.compatible') : $t('plugins.incompatible') }}
                  </el-tag>
                </div>
              </div>
              <div class="plugin-desc">{{ tr(item.description) || $t('plugins.noDescription') }}</div>
              <div class="plugin-tags">
                <el-tag v-for="tag in item.tags || []" :key="tag" size="small" type="info" effect="plain">{{ tr(tag) }}</el-tag>
                <el-tag v-for="permission in item.permissions || []" :key="permission" size="small" type="warning" effect="plain">{{ $t('plugins.permissionTag', { name: permission }) }}</el-tag>
              </div>
              <div class="pipeline-list">
                <div v-for="pipeline in item.pipelines || []" :key="pipeline.id" class="pipeline-row">
                  <span>{{ tr(pipeline.name) }}</span>
                  <span class="text-muted">{{ pipeline.paradigm }}</span>
                  <span class="mono text-muted" style="font-size:11px">{{ pipeline.id }}</span>
                </div>
              </div>
              <div class="plugin-foot">
                <el-tag v-if="item.installed" size="small" type="success" effect="plain">{{ $t('plugins.installedVersion', { version: item.installed_version }) }}</el-tag>
                <el-button
                  v-else
                  type="primary"
                  size="small"
                  :disabled="!admin || !item.compatible"
                  :title="item.compatible ? '' : item.compatibility"
                  :loading="busy === item.id"
                  @click="installBundled(item)"
                >{{ $t('plugins.installEnable') }}</el-button>
                <span class="text-muted" style="font-size:12px">{{ item.author || 'NeuroKiln' }} · {{ item.license || $t('plugins.licenseUndeclared') }}</span>
              </div>
            </el-card>
          </div>
        </el-tab-pane>

        <el-tab-pane :label="$t('plugins.installedTab', { count: plugins.length })" name="installed">
          <el-table :data="plugins" border stripe v-loading="loading">
            <el-table-column :label="$t('plugins.pluginColumn')" min-width="220">
              <template #default="{ row }">
                <el-link type="primary" @click="openDetail(row)">{{ tr(row.name) }}</el-link>
                <div class="mono text-muted" style="font-size:12px">{{ row.id }} · v{{ row.version }}</div>
              </template>
            </el-table-column>
            <el-table-column :label="$t('plugins.runtimePermissionsColumn')" width="170" align="center">
              <template #default="{ row }">
                <el-tag size="small" :type="row.runtime === 'python' ? 'warning' : 'success'" effect="plain">{{ row.runtime }}</el-tag>
                <div style="margin-top:4px">
                  <el-tag v-for="permission in row.permissions || []" :key="permission" size="small" type="info" effect="plain" style="margin-right:4px">{{ permission }}</el-tag>
                </div>
              </template>
            </el-table-column>
            <el-table-column :label="$t('plugins.manifestCompatibilityColumn')" width="160" align="center">
              <template #default="{ row }">
                <el-tag size="small" type="info" effect="plain">v{{ row.schema_version }}</el-tag>
                <el-tag
                  size="small"
                  :type="row.compatible ? 'success' : 'danger'"
                  effect="plain"
                  style="margin-left:4px"
                >{{ row.compatible ? $t('plugins.compatible') : $t('plugins.incompatible') }}</el-tag>
                <div class="text-muted" style="font-size:11px;margin-top:4px">
                  {{ row.engine_min_version ? $t('plugins.engineMinVersion', { version: row.engine_min_version }) : $t('plugins.legacyManifest') }}
                </div>
              </template>
            </el-table-column>
            <el-table-column :label="$t('plugins.algorithmsColumn')" min-width="200">
              <template #default="{ row }">
                <div v-for="pipeline in row.pipelines || []" :key="pipeline.id" class="pipeline-row">
                  <span>{{ tr(pipeline.name) }}</span>
                  <el-tag size="small" type="info" effect="plain">{{ pipeline.paradigm }}</el-tag>
                </div>
                <span v-if="!row.pipelines?.length" class="text-muted">-</span>
              </template>
            </el-table-column>
            <el-table-column :label="$t('plugins.statusColumn')" width="150" align="center">
              <template #default="{ row }">
                <el-tag v-if="row.enabled" size="small" type="success">{{ $t('plugins.enabledLabel') }}</el-tag>
                <el-tag v-else-if="row.error" size="small" type="danger">{{ $t('plugins.errorLabel') }}</el-tag>
                <el-tag v-else size="small" type="info">{{ $t('plugins.disabledLabel') }}</el-tag>
                <div class="text-muted" style="font-size:11px;margin-top:4px">
                  {{ row.verified ? $t('plugins.hashVerified') : (row.requires_trust ? $t('plugins.requiresTrust') : $t('plugins.inactive')) }}
                </div>
              </template>
            </el-table-column>
            <el-table-column :label="$t('plugins.sourceInstalledColumn')" min-width="170">
              <template #default="{ row }">
                <div>{{ row.source || '-' }}</div>
                <div class="text-muted" style="font-size:12px">{{ formatTime(row.installed_at) }}</div>
              </template>
            </el-table-column>
            <el-table-column :label="$t('plugins.actionsColumn')" width="220" align="center" fixed="right">
              <template #default="{ row }">
                <el-button link type="primary" @click="openDetail(row)">{{ $t('plugins.detail') }}</el-button>
                <el-button
                  v-if="!row.enabled"
                  link
                  type="success"
                  :disabled="!admin || !row.compatible || !row.verified && row.requires_trust && !status.trusted"
                  :loading="busy === row.id"
                  @click="toggle(row, 'enable')"
                >{{ $t('plugins.enableAction') }}</el-button>
                <el-button v-else link type="warning" :disabled="!admin" :loading="busy === row.id" @click="toggle(row, 'disable')">{{ $t('plugins.disableAction') }}</el-button>
                <el-button link type="danger" :disabled="!admin" :loading="busy === row.id" @click="remove(row)">{{ $t('plugins.uninstallAction') }}</el-button>
              </template>
            </el-table-column>
            <template #empty>
              <el-empty :description="$t('plugins.installedEmpty')" />
            </template>
          </el-table>
        </el-tab-pane>
      </el-tabs>
    </el-card>

    <el-dialog v-model="uploadVisible" :title="$t('plugins.uploadZip')" width="540px" @closed="resetUpload">
      <el-alert
        type="info"
        :closable="false"
        style="margin-bottom:12px"
        :title="$t('plugins.uploadHint')"
      />
      <el-upload
        ref="uploadRef"
        drag
        :auto-upload="false"
        :limit="1"
        accept=".zip"
        :on-change="(file) => (uploadFile = file.raw)"
        :on-remove="() => (uploadFile = null)"
        style="width:100%"
      >
        <el-icon class="el-icon--upload"><Upload /></el-icon>
        <div class="el-upload__text">{{ $t('plugins.uploadDropPrefix') }}<em>{{ $t('plugins.uploadDropAction') }}</em></div>
      </el-upload>
      <el-checkbox v-model="uploadEnable" style="margin-top:12px">{{ $t('plugins.enableAfterInstall') }}</el-checkbox>
      <template #footer>
        <el-button @click="uploadVisible = false">{{ $t('common.cancel') }}</el-button>
        <el-button type="primary" :loading="uploading" @click="submitUpload">{{ $t('plugins.uploadAndVerify') }}</el-button>
      </template>
    </el-dialog>

    <el-drawer v-model="detailVisible" :title="detail ? `${tr(detail.name)} · ${detail.id}` : $t('plugins.detailTitle')" size="min(720px, 94%)">
      <template v-if="detail">
        <el-alert
          v-if="detail.error"
          type="error"
          :closable="false"
          style="margin-bottom:12px"
          :title="$t('plugins.pluginError', { error: detail.error })"
        />
        <el-descriptions :column="2" border size="small">
          <el-descriptions-item :label="$t('plugins.version')">{{ detail.version }}</el-descriptions-item>
          <el-descriptions-item :label="$t('plugins.runtime')">{{ detail.runtime }}</el-descriptions-item>
          <el-descriptions-item :label="$t('plugins.schemaLabel')">{{ $t('plugins.schemaNodeApi', { schema: detail.schema_version, api: detail.api_version }) }}</el-descriptions-item>
          <el-descriptions-item :label="$t('plugins.compatibility')">
            <el-tag size="small" :type="detail.compatible ? 'success' : 'danger'">{{ detail.compatibility || '-' }}</el-tag>
          </el-descriptions-item>
          <el-descriptions-item :label="$t('plugins.engineRequirement')">{{ detail.engine_min_version || $t('plugins.engineUndeclared') }}</el-descriptions-item>
          <el-descriptions-item :label="$t('plugins.statusColumn')">
            <el-tag size="small" :type="detail.enabled ? 'success' : 'info'">{{ detail.enabled ? $t('plugins.enabledLabel') : $t('plugins.disabledLabel') }}</el-tag>
          </el-descriptions-item>
          <el-descriptions-item :label="$t('plugins.integrity')">
            <el-tag size="small" :type="detail.verified ? 'success' : 'warning'">{{ detail.verified ? $t('plugins.verified') : $t('plugins.inactive') }}</el-tag>
          </el-descriptions-item>
          <el-descriptions-item :label="$t('plugins.author')">{{ detail.author || '-' }}</el-descriptions-item>
          <el-descriptions-item :label="$t('plugins.license')">{{ detail.license || '-' }}</el-descriptions-item>
          <el-descriptions-item :label="$t('plugins.permissions')" :span="2">
            <el-tag v-for="permission in detail.permissions || []" :key="permission" size="small" type="warning" effect="plain" style="margin-right:6px">{{ permission }}</el-tag>
            <span v-if="!detail.permissions?.length" class="text-muted">{{ $t('plugins.none') }}</span>
          </el-descriptions-item>
          <el-descriptions-item :label="$t('plugins.packageHash')" :span="2"><span class="mono hash">{{ detail.package_sha256 || '-' }}</span></el-descriptions-item>
          <el-descriptions-item :label="$t('plugins.manifestHash')" :span="2"><span class="mono hash">{{ detail.manifest_sha256 || '-' }}</span></el-descriptions-item>
          <el-descriptions-item :label="$t('plugins.source')">{{ detail.source || '-' }}</el-descriptions-item>
          <el-descriptions-item :label="$t('plugins.installedAt')">{{ formatTime(detail.installed_at) }}</el-descriptions-item>
          <el-descriptions-item :label="$t('plugins.descriptionLabel')" :span="2">{{ tr(detail.description) || '-' }}</el-descriptions-item>
        </el-descriptions>

        <div class="section-title" style="margin-top:16px">{{ $t('plugins.providedPipelines') }}</div>
        <el-table :data="detail.pipelines || []" border size="small">
          <el-table-column :label="$t('plugins.name')" min-width="150">
            <template #default="{ row }">{{ tr(row.name) }}</template>
          </el-table-column>
          <el-table-column prop="paradigm" :label="$t('plugins.paradigm')" width="100" align="center" />
          <el-table-column prop="id" :label="$t('plugins.pipelineId')" min-width="220">
            <template #default="{ row }"><span class="mono">{{ row.id }}</span></template>
          </el-table-column>
          <el-table-column :label="$t('plugins.descriptionLabel')" min-width="200">
            <template #default="{ row }">{{ tr(row.description) }}</template>
          </el-table-column>
        </el-table>

        <div class="section-title" style="margin-top:16px">{{ $t('plugins.packageFiles', { count: detail.files?.length || 0 }) }}</div>
        <div class="file-list">
          <div v-for="file in detail.files || []" :key="file" class="mono">{{ file }}</div>
          <span v-if="!detail.files?.length" class="text-muted">{{ $t('plugins.noFiles') }}</span>
        </div>

        <el-alert
          v-if="detail.requires_trust"
          type="warning"
          :closable="false"
          style="margin-top:16px"
          :title="$t('plugins.pythonTrustWarning')"
        />
      </template>
    </el-drawer>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { tr } from '@/i18n/catalog'
import { ElMessage, ElMessageBox } from 'element-plus'
import {
  disablePlugin,
  enablePlugin,
  getPlugins,
  installBundledPlugin,
  installPluginZip,
  uninstallPlugin,
} from '@/api'
import { isAdmin } from '@/utils/auth'
import { formatTime } from '@/utils/format'

const { t } = useI18n()
const status = ref({})
const plugins = computed(() => status.value.plugins || [])
const catalog = computed(() => status.value.catalog || [])
const admin = computed(() => isAdmin())
const loading = ref(false)
const tab = ref('catalog')
const busy = ref('')
const detail = ref(null)
const detailVisible = ref(false)
const uploadVisible = ref(false)
const uploading = ref(false)
const uploadFile = ref(null)
const uploadEnable = ref(true)
const uploadRef = ref(null)

async function refresh() {
  loading.value = true
  try {
    status.value = await getPlugins()
  } finally {
    loading.value = false
  }
}

function openDetail(row) {
  detail.value = row
  detailVisible.value = true
}

async function installBundled(item) {
  busy.value = item.id
  try {
    await installBundledPlugin({ id: item.id, enable: true })
    ElMessage.success(t('plugins.installedEnabled', { name: item.name }))
    await refresh()
  } finally {
    busy.value = ''
  }
}

async function toggle(row, action) {
  busy.value = row.id
  try {
    if (action === 'enable') await enablePlugin(row.id)
    else await disablePlugin(row.id)
    ElMessage.success(action === 'enable' ? t('plugins.enabledMessage') : t('plugins.disabledMessage'))
    await refresh()
    if (detail.value?.id === row.id) {
      detail.value = plugins.value.find((item) => item.id === row.id) || detail.value
    }
  } finally {
    busy.value = ''
  }
}

async function remove(row) {
  try {
    await ElMessageBox.confirm(
      t('plugins.uninstallConfirm', { name: row.name }),
      t('plugins.uninstallTitle'),
      { type: 'warning', confirmButtonText: t('plugins.uninstallAction'), cancelButtonText: t('common.cancel') },
    )
  } catch {
    return
  }
  busy.value = row.id
  try {
    await uninstallPlugin(row.id)
    ElMessage.success(t('plugins.uninstalledMessage'))
    detailVisible.value = false
    await refresh()
  } finally {
    busy.value = ''
  }
}

function openUpload() {
  uploadVisible.value = true
}

function resetUpload() {
  uploadRef.value?.clearFiles()
  uploadFile.value = null
  uploadEnable.value = true
}

async function submitUpload() {
  if (!uploadFile.value) {
    ElMessage.warning(t('plugins.selectZipFile'))
    return
  }
  uploading.value = true
  try {
    const formData = new FormData()
    formData.append('file', uploadFile.value)
    formData.append('enable', String(uploadEnable.value))
    const result = await installPluginZip(formData)
    ElMessage.success(result.enabled
      ? t('plugins.uploadedEnabledMessage', { id: result.id })
      : t('plugins.uploadedMessage', { id: result.id }))
    uploadVisible.value = false
    await refresh()
    tab.value = 'installed'
  } finally {
    uploading.value = false
  }
}

onMounted(refresh)
</script>

<style scoped>
.plugin-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 16px; }
.plugin-card { display: flex; flex-direction: column; }
.plugin-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 8px; }
.plugin-badges { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 4px; max-width: 170px; }
.plugin-name { font-weight: 600; font-size: 15px; margin-bottom: 2px; }
.plugin-desc { color: #606266; font-size: 13px; line-height: 1.7; margin: 10px 0; min-height: 44px; }
.plugin-tags { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
.pipeline-list { border-top: 1px dashed #ebeef5; padding-top: 8px; }
.pipeline-row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; font-size: 12px; line-height: 1.9; }
.plugin-foot { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; justify-content: space-between; margin-top: 12px; }
.section-title { font-size: 13px; font-weight: 600; margin-bottom: 8px; }
.hash { word-break: break-all; font-size: 12px; }
.file-list { border: 1px solid #ebeef5; border-radius: 6px; padding: 10px 12px; font-size: 12px; line-height: 1.9; max-height: 180px; overflow: auto; background: #fafafa; }
</style>
