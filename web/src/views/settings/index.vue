<template>
  <div class="app-container">
    <h2 class="page-title">{{ $t('settings.title') }}</h2>
    <p class="page-desc">{{ $t('settings.description') }}</p>
    <el-row :gutter="16">
      <el-col :xs="24" :md="24" :lg="12" style="margin-bottom:16px">
        <el-card shadow="never">
          <template #header><b>{{ $t('settings.environment') }}</b></template>
          <el-descriptions :column="1" border>
            <el-descriptions-item :label="$t('settings.apiStatus')">
              <el-tag :type="health.status === 'ok' ? 'success' : 'danger'" size="small">{{ health.status || 'unknown' }}</el-tag>
            </el-descriptions-item>
            <el-descriptions-item :label="$t('settings.apiVersion')">{{ health.api || '-' }}</el-descriptions-item>
            <el-descriptions-item :label="$t('settings.engineVersion')">{{ health.engine || '-' }}</el-descriptions-item>
            <el-descriptions-item :label="$t('settings.enginePath')"><span class="mono" style="font-size:12px">{{ health.engine_path || '-' }}</span></el-descriptions-item>
            <el-descriptions-item :label="$t('settings.dataRoot')"><span class="mono" style="font-size:12px">{{ health.data_root || '-' }}</span></el-descriptions-item>
            <el-descriptions-item :label="$t('settings.sqliteDatabase')"><span class="mono" style="font-size:12px">{{ health.database || '-' }}</span></el-descriptions-item>
            <el-descriptions-item :label="$t('settings.moabbData')">
              <el-tag :type="health.moabb_installed ? 'success' : 'warning'" size="small">{{ health.moabb_installed ? $t('settings.installed') : $t('settings.notInstalled') }}</el-tag>
            </el-descriptions-item>
            <el-descriptions-item :label="$t('settings.torchBaseline')">
              <el-tag :type="health.torch_available ? 'success' : 'info'" size="small">{{ health.torch_available ? $t('settings.available') : $t('settings.notInstalled') }}</el-tag>
            </el-descriptions-item>
            <el-descriptions-item :label="$t('settings.customPipelines')">{{ health.custom_pipelines ?? '-' }}</el-descriptions-item>
            <el-descriptions-item :label="$t('settings.apiDocs')"><el-link type="primary" href="http://127.0.0.1:8100/docs" target="_blank">/docs</el-link></el-descriptions-item>
          </el-descriptions>
        </el-card>
      </el-col>
      <el-col :xs="24" :md="24" :lg="12">
        <el-card shadow="never">
          <template #header><b>{{ $t('settings.experimentDefaults') }}</b></template>
          <el-form label-width="150px">
            <el-form-item :label="$t('settings.globalSeed')"><el-input-number v-model="seed" :min="0" :max="999999" @change="save" /></el-form-item>
            <el-form-item :label="$t('settings.dataRoot')"><el-input v-model="dataDir" placeholder="D:/neurokiln-data" @change="save" /></el-form-item>
            <el-form-item :label="$t('settings.strictLeakage')"><el-switch :model-value="true" disabled :active-text="$t('settings.forceEnabled')" /></el-form-item>
            <el-form-item :label="$t('settings.thirdPartyPlugins')">
              <el-tag size="small" type="info" effect="plain">{{ $t('settings.thirdPartyPluginsHint') }}</el-tag>
            </el-form-item>
            <el-form-item :label="$t('settings.pythonPlugins')">
              <el-tag size="small" :type="pluginMode === 'trusted' ? 'warning' : 'success'" effect="plain">
                {{ pluginMode === 'trusted' ? $t('settings.trustedMode') : $t('settings.restrictedMode') }}
              </el-tag>
            </el-form-item>
          </el-form>
        </el-card>
      </el-col>
    </el-row>

    <el-card shadow="never" style="margin-top:16px">
      <template #header>
        <div class="section-head">
          <b>{{ $t('settings.cacheTitle') }}</b>
          <div>
            <el-button size="small" icon="Refresh" :loading="cacheLoading" @click="loadCache">{{ $t('common.refresh') }}</el-button>
            <el-button size="small" type="danger" plain icon="Delete" :disabled="!admin || cacheLoading" @click="onClearCache">{{ $t('settings.clearCache') }}</el-button>
          </div>
        </div>
      </template>
      <el-descriptions :column="3" border size="small">
        <el-descriptions-item :label="$t('settings.status')">
          <el-tag size="small" :type="cache.enabled ? 'success' : 'info'">{{ cache.enabled ? $t('settings.enabled') : $t('settings.disabled') }}</el-tag>
        </el-descriptions-item>
        <el-descriptions-item :label="$t('settings.entriesUsage')">{{ cache.entries ?? '-' }} / {{ formatBytes(cache.bytes) }}</el-descriptions-item>
        <el-descriptions-item :label="$t('settings.limit')">{{ formatBytes(cache.max_bytes) }} · {{ $t('settings.entriesCount', { count: cache.max_entries ?? '-' }) }}</el-descriptions-item>
        <el-descriptions-item :label="$t('settings.hitsMisses')">{{ cache.hits ?? '-' }} / {{ cache.misses ?? '-' }}</el-descriptions-item>
        <el-descriptions-item :label="$t('settings.hitRate')">{{ cache.hit_rate === undefined ? '-' : `${(cache.hit_rate * 100).toFixed(1)}%` }}</el-descriptions-item>
        <el-descriptions-item :label="$t('settings.writesEvictions')">{{ cache.writes ?? '-' }} / {{ cache.evictions ?? '-' }}</el-descriptions-item>
        <el-descriptions-item :label="$t('settings.directory')" :span="3"><span class="mono" style="font-size:12px">{{ cache.root || '-' }}</span></el-descriptions-item>
      </el-descriptions>
      <div class="text-muted" style="font-size:12px;line-height:1.8;margin-top:8px">
        {{ $t('settings.cacheNoteLine1') }}
        {{ $t('settings.cacheNoteEnabledPrefix') }} <span class="mono">--no-cache</span> / <span class="mono">use_cache=false</span> {{ $t('settings.cacheNoteEnabledSuffix') }}
      </div>
    </el-card>

    <el-card shadow="never" style="margin-top:16px">
      <template #header>
        <div class="section-head">
          <b>{{ $t('settings.accountPermissions') }}</b>
          <el-button v-if="admin" type="primary" size="small" icon="Plus" @click="openCreate">{{ $t('settings.createAccount') }}</el-button>
        </div>
      </template>
      <el-descriptions :column="3" border size="small">
        <el-descriptions-item :label="$t('settings.currentAccount')">{{ authState.user?.username || '-' }}</el-descriptions-item>
        <el-descriptions-item :label="$t('settings.role')">
          <el-tag size="small" :type="admin ? 'warning' : 'info'" effect="plain">{{ admin ? $t('settings.admin') : $t('settings.member') }}</el-tag>
        </el-descriptions-item>
        <el-descriptions-item :label="$t('settings.tokenExpiry')">{{ formatTime(expiresAt) }}</el-descriptions-item>
      </el-descriptions>

      <template v-if="admin">
        <el-table :data="users" border stripe size="small" style="margin-top:14px" v-loading="usersLoading">
          <el-table-column prop="id" label="#" width="60" align="center" />
          <el-table-column prop="username" :label="$t('settings.username')" min-width="150" />
          <el-table-column :label="$t('settings.role')" width="100" align="center">
            <template #default="{ row }">
              <el-tag size="small" :type="row.role === 'admin' ? 'warning' : 'info'" effect="plain">
                {{ row.role === 'admin' ? $t('settings.admin') : $t('settings.member') }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column :label="$t('settings.createdAt')" width="180" align="center">
            <template #default="{ row }">{{ formatTime(row.created_at) }}</template>
          </el-table-column>
          <el-table-column :label="$t('settings.lastLogin')" width="180" align="center">
            <template #default="{ row }">{{ formatTime(row.last_login_at) }}</template>
          </el-table-column>
          <el-table-column :label="$t('settings.actions')" width="100" align="center">
            <template #default="{ row }">
              <el-button
                link
                type="danger"
                :disabled="row.id === authState.user?.id"
                @click="removeUser(row)"
              >{{ $t('common.delete') }}</el-button>
            </template>
          </el-table-column>
        </el-table>
      </template>
      <el-empty v-else :image-size="70" :description="$t('settings.memberEmpty')" />
    </el-card>

    <el-dialog v-model="userDialog" :title="$t('settings.createAccount')" width="440px">
      <el-form ref="userFormRef" :model="userForm" :rules="userRules" label-width="80px">
        <el-form-item :label="$t('settings.username')" prop="username">
          <el-input v-model="userForm.username" :placeholder="$t('settings.usernamePlaceholder')" />
        </el-form-item>
        <el-form-item :label="$t('settings.password')" prop="password">
          <el-input v-model="userForm.password" type="password" show-password :placeholder="$t('settings.passwordPlaceholder')" />
        </el-form-item>
        <el-form-item :label="$t('settings.role')" prop="role">
          <el-select v-model="userForm.role" style="width:100%">
            <el-option :label="$t('settings.member')" value="member" />
            <el-option :label="$t('settings.admin')" value="admin" />
          </el-select>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="userDialog = false">{{ $t('common.cancel') }}</el-button>
        <el-button type="primary" :loading="creatingUser" @click="submitUser">{{ $t('settings.create') }}</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { computed, onMounted, reactive, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { ElMessage, ElMessageBox } from 'element-plus'
import { clearCache, createUser, deleteUser, getCacheStats, getHealth, getPlugins, getUsers } from '@/api'
import { authState, isAdmin } from '@/utils/auth'
import { formatTime } from '@/utils/format'

const { t } = useI18n()
const health = ref({})
const cache = ref({})
const cacheLoading = ref(false)
const seed = ref(0)
const dataDir = ref('D:/neurokiln-data')
const pluginMode = ref('restricted')
const users = ref([])
const usersLoading = ref(false)
const userDialog = ref(false)
const creatingUser = ref(false)
const userFormRef = ref(null)
const userForm = reactive({ username: '', password: '', role: 'member' })

const admin = computed(() => isAdmin())
const expiresAt = computed(() => (
  authState.expiresAt ? new Date(authState.expiresAt * 1000).toISOString() : null
))
const userRules = computed(() => ({
  username: [
    { required: true, message: t('settings.usernameRequired'), trigger: 'blur' },
    { pattern: /^[A-Za-z0-9_.-]{3,32}$/, message: t('settings.usernameRule'), trigger: 'blur' },
  ],
  password: [
    { required: true, message: t('settings.passwordRequired'), trigger: 'blur' },
    { min: 8, message: t('settings.passwordMin'), trigger: 'blur' },
  ],
}))

function save() {
  localStorage.setItem('neurokiln-settings', JSON.stringify({
    seed: seed.value,
    dataDir: dataDir.value,
  }))
  ElMessage.success(t('settings.savedMessage'))
}

function formatBytes(value) {
  if (value === null || value === undefined) return '-'
  const units = ['B', 'KiB', 'MiB', 'GiB']
  let size = Number(value)
  let index = 0
  while (size >= 1024 && index < units.length - 1) {
    size /= 1024
    index += 1
  }
  return `${size.toFixed(index === 0 ? 0 : 1)} ${units[index]}`
}

async function loadCache() {
  cacheLoading.value = true
  try {
    cache.value = await getCacheStats()
  } finally {
    cacheLoading.value = false
  }
}

async function onClearCache() {
  try {
    await ElMessageBox.confirm(t('settings.clearCacheConfirm'), t('settings.clearCache'), {
      type: 'warning', confirmButtonText: t('settings.clearConfirm'), cancelButtonText: t('common.cancel'),
    })
  } catch {
    return
  }
  cacheLoading.value = true
  try {
    const result = await clearCache()
    cache.value = result
    ElMessage.success(t('settings.cacheCleared'))
  } finally {
    cacheLoading.value = false
  }
}

async function loadUsers() {
  if (!admin.value) return
  usersLoading.value = true
  try {
    users.value = (await getUsers()).items
  } finally {
    usersLoading.value = false
  }
}

function openCreate() {
  userForm.username = ''
  userForm.password = ''
  userForm.role = 'member'
  userDialog.value = true
}

async function submitUser() {
  try {
    await userFormRef.value.validate()
  } catch {
    return
  }
  creatingUser.value = true
  try {
    await createUser({
      username: userForm.username.trim(),
      password: userForm.password,
      role: userForm.role,
    })
    ElMessage.success(t('settings.accountCreated'))
    userDialog.value = false
    await loadUsers()
  } finally {
    creatingUser.value = false
  }
}

async function removeUser(row) {
  try {
    await ElMessageBox.confirm(t('settings.deleteAccountConfirm', { name: row.username }), t('settings.deleteAccountTitle'), {
      type: 'warning', confirmButtonText: t('common.delete'), cancelButtonText: t('common.cancel'),
    })
  } catch {
    return
  }
  await deleteUser(row.id)
  ElMessage.success(t('settings.accountDeleted'))
  await loadUsers()
}

onMounted(async () => {
  health.value = await getHealth()
  await loadCache()
  try {
    pluginMode.value = (await getPlugins()).mode || 'restricted'
  } catch {
    pluginMode.value = 'restricted'
  }
  const raw = localStorage.getItem('neurokiln-settings')
  if (raw) {
    const saved = JSON.parse(raw)
    seed.value = saved.seed ?? 0
    dataDir.value = saved.dataDir ?? dataDir.value
  }
  await loadUsers()
})
</script>

<style scoped>
.section-head { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; justify-content: space-between; }
</style>
