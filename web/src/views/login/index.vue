<template>
  <div class="login-page">
    <div class="login-panel">
      <div class="brand">
        <svg viewBox="0 0 32 32" width="40" height="40" aria-hidden="true">
          <path d="M12 3h8v3h-2v6.2l7.4 12.1A3 3 0 0 1 22.8 29H9.2a3 3 0 0 1-2.6-4.7L14 12.2V6h-2V3z" fill="#409eff"/>
          <path d="M10.5 21h11l2.2 3.6a1.2 1.2 0 0 1-1 1.9H9.3a1.2 1.2 0 0 1-1-1.9L10.5 21z" fill="#a0cfff"/>
        </svg>
        <div>
          <h1>NeuroKiln 脑炉</h1>
          <p>{{ $t('login.subtitle') }}</p>
        </div>
      </div>

      <el-card shadow="never" class="login-card">
        <div class="mode-title">
          <h2>{{ setup ? $t('login.setupTitle') : $t('login.loginTitle') }}</h2>
          <p>{{ setup ? $t('login.setupHint') : $t('login.loginHint') }}</p>
        </div>

        <el-alert
          v-if="setup"
          type="warning"
          :closable="false"
          :title="$t('login.setupWarning')"
          style="margin-bottom:18px"
        />

        <el-form ref="formRef" :model="form" :rules="rules" label-position="top" @submit.prevent="submit">
          <el-form-item :label="$t('login.username')" prop="username">
            <el-input
              v-model="form.username"
              size="large"
              :placeholder="$t('login.usernamePlaceholder')"
              :prefix-icon="User"
              autocomplete="username"
            />
          </el-form-item>
          <el-form-item :label="$t('login.password')" prop="password">
            <el-input
              v-model="form.password"
              size="large"
              type="password"
              show-password
              :placeholder="$t('login.passwordPlaceholder')"
              :prefix-icon="Lock"
              :autocomplete="setup ? 'new-password' : 'current-password'"
              @keyup.enter="submit"
            />
          </el-form-item>
          <el-form-item v-if="setup" :label="$t('login.confirm')" prop="confirm">
            <el-input
              v-model="form.confirm"
              size="large"
              type="password"
              show-password
              :prefix-icon="Lock"
              autocomplete="new-password"
              @keyup.enter="submit"
            />
          </el-form-item>
          <el-button type="primary" size="large" class="submit" :loading="loading" @click="submit">
            {{ setup ? $t('login.create') : $t('login.submit') }}
          </el-button>
        </el-form>

        <div class="login-foot">
          <span>{{ $t('login.footer') }}</span>
          <el-link type="primary" href="http://127.0.0.1:8100/docs" target="_blank">{{ $t('login.apiDocs') }}</el-link>
        </div>
      </el-card>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, reactive, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { Lock, User } from '@element-plus/icons-vue'
import { bootstrapAdmin, getAuthStatus, login } from '@/api'
import { setSession } from '@/utils/auth'

const { t } = useI18n()
const route = useRoute()
const router = useRouter()
const formRef = ref(null)
const loading = ref(false)
const setup = ref(false)
const form = reactive({ username: '', password: '', confirm: '' })

const rules = computed(() => ({
  username: [
    { required: true, message: t('login.usernameRequired'), trigger: 'blur' },
    { pattern: /^[A-Za-z0-9_.-]{3,32}$/, message: t('login.usernameRule'), trigger: 'blur' },
  ],
  password: [
    { required: true, message: t('login.passwordRequired'), trigger: 'blur' },
    { min: 8, message: t('login.passwordMin'), trigger: 'blur' },
  ],
  ...(setup.value
    ? {
        confirm: [
          { required: true, message: t('login.confirmRequired'), trigger: 'blur' },
          {
            validator: (_rule, value, callback) => {
              if (value === form.password) callback()
              else callback(new Error(t('login.passwordMismatch')))
            },
            trigger: 'blur',
          },
        ],
      }
    : {}),
}))

function targetPath() {
  const redirect = route.query.redirect
  if (typeof redirect === 'string' && redirect.startsWith('/') && !redirect.startsWith('//')) {
    return redirect
  }
  return '/dashboard'
}

async function submit() {
  if (loading.value) return
  try {
    await formRef.value.validate()
  } catch {
    return
  }
  loading.value = true
  try {
    const payload = { username: form.username.trim(), password: form.password }
    const session = setup.value ? await bootstrapAdmin(payload) : await login(payload)
    setSession(session)
    ElMessage.success(setup.value ? t('login.created') : t('login.welcome', { name: session.user.username }))
    await router.replace(targetPath())
  } finally {
    loading.value = false
  }
}

onMounted(async () => {
  try {
    const status = await getAuthStatus()
    setup.value = !status.initialized
  } catch {
    setup.value = false
  }
})
</script>

<style scoped>
.login-page {
  min-height: 100%; display: flex; align-items: center; justify-content: center;
  padding: 32px 20px; position: relative; overflow: hidden;
  background:
    radial-gradient(circle at 12% 18%, rgba(64, 158, 255, .30), transparent 34%),
    radial-gradient(circle at 86% 82%, rgba(103, 194, 58, .16), transparent 32%),
    linear-gradient(135deg, #101a2e 0%, #172b4d 52%, #0d2338 100%);
}
.login-page::before {
  content: ""; position: absolute; inset: 0; pointer-events: none;
  background-image:
    linear-gradient(rgba(160, 207, 255, .055) 1px, transparent 1px),
    linear-gradient(90deg, rgba(160, 207, 255, .055) 1px, transparent 1px);
  background-size: 46px 46px;
  mask-image: radial-gradient(circle at 50% 40%, #000 20%, transparent 78%);
}
.login-panel { width: min(460px, 100%); position: relative; z-index: 1; }
.brand {
  display: flex; align-items: center; gap: 14px; color: #eaf3ff;
  margin-bottom: 20px; padding: 0 6px;
}
.brand h1 { margin: 0; font-size: 22px; letter-spacing: .5px; }
.brand p { margin: 5px 0 0; color: #9db8d9; font-size: 13px; }
.login-card {
  border: 1px solid rgba(255, 255, 255, .12); border-radius: 14px;
  box-shadow: 0 26px 60px rgba(0, 12, 32, .46);
  background: rgba(255, 255, 255, .97);
}
.mode-title { margin-bottom: 18px; }
.mode-title h2 { margin: 0; font-size: 20px; color: #1f2d3d; }
.mode-title p { margin: 6px 0 0; color: #909399; font-size: 13px; }
.submit { width: 100%; margin-top: 4px; letter-spacing: 2px; }
.login-foot {
  display: flex; align-items: center; justify-content: space-between;
  margin-top: 20px; padding-top: 14px; border-top: 1px solid #ebeef5;
  color: #a8abb2; font-size: 12px;
}
</style>
