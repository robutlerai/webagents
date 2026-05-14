export {
  CustomHttpSkill,
  type CustomHttpSkillConfig,
  type CustomHttpEndpointEntry,
} from './skill';

// Re-export the templates registry so portal-side knowledge surfaces
// (FactoryKnowledgeSkill / get_doc) can synthesize per-template
// markdown without each consumer reaching deep into the templates
// folder.
export {
  WEBAPP_TEMPLATES,
  WEBAPP_TEMPLATE_NAMES,
  getWebappTemplate,
  listWebappTemplates,
  type WebappTemplate,
  type WebappTemplateName,
  type RequiredPermissions,
} from './templates';
