"""Module for handling memory configuration and templating.

This module provides classes to read and process the memory configuration file,
which includes memory types, section categories, and sections with their schemas.
It uses Pydantic for data validation and Jinja2 for template rendering.
"""

from typing import List, Optional, Dict, Any, Literal, Type
import yaml
import logging
from pydantic import BaseModel, Field, computed_field, create_model
from jinja2 import Environment, BaseLoader
from importlib.resources import files


logger = logging.getLogger(__name__)


class MemoryType(BaseModel):
    """Represents a memory type definition.

    Attributes:
        id: Unique identifier for the memory type.
        name: Display name of the memory type.
        description: Detailed description of what this memory type represents.
    """

    id: str = Field(description="Unique identifier for the memory type")
    name: str = Field(description="Display name of the memory type")
    description: str = Field(description="Detailed description of what this memory type represents")


class SectionCategory(BaseModel):
    """Represents a section category definition.

    Attributes:
        id: Unique identifier for the category.
        name: Display name of the category.
        description: Detailed description of what this category represents.
        written_by: Who writes to this category (user or system).
    """

    id: str = Field(description="Unique identifier for the section category")
    name: str = Field(description="Display name of the section category")
    description: str = Field(description="Detailed description of what this section category represents")
    written_by: Literal["user", "system"] = Field(description="Who writes to this section category (user or system)")


class SuggestedProperty(BaseModel):
    """Represents a suggested property in a memory schema.

    Attributes:
        name: Name of the property.
        description: Description of what this property represents.
        type: Optional type of the property (defaults to string, can be list).
        update_method: How to update this property (replace or append).
    """

    name: str = Field(description="Name of the property")
    description: str = Field(description="Description of what this property represents")
    type: Literal["string", "list"] = Field(
        default="string", description="Type of the property (defaults to string, can be list)"
    )
    update_method: Literal["replace", "append"] = Field(
        default="replace", description="How to update this property (replace or append)"
    )
    context_window_prefix: Optional[str] = Field(
        default=None, description="Prefix to use in the context window for the property"
    )
    embedding_prefix: Optional[str] = Field(
        default=None, description="Prefix to use in the embedding for the property"
    )


class Schema(BaseModel):
    """Represents a memory schema definition.

    Attributes:
        suggested_properties: List of suggested properties for this schema.
        corpus: Optional corpus configuration for knowledge-based schemas.
    """

    suggested_properties: Optional[List[SuggestedProperty]] = Field(
        default=None, description="List of suggested properties for this schema"
    )
    corpus: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional corpus configuration for knowledge-based schemas"
    )


class Section(BaseModel):
    """Represents a memory section definition.

    Attributes:
        id: Unique identifier for the section.
        name: Display name of the section.
        description: Detailed description of what this section represents.
        category: Category this section belongs to.
        type: Memory type this section represents.
        memory_schema: Schema definition for this section.
    """

    id: str = Field(description="Unique identifier for the section")
    name: str = Field(description="Display name of the section")
    description: str = Field(description="Detailed description of what this section represents")
    category: str = Field(description="Category this section belongs to")
    type: str = Field(description="Memory type this section represents")
    memory_schema: Schema = Field(alias="schema")

    def get_property(self, property_id: str) -> Optional[SuggestedProperty]:
        """
        Returns the property with the given ID.

        Args:
            property_id: The ID of the property to get.

        Returns:
            SuggestedProperty: The property with the given ID.
        """
        if self.memory_schema.suggested_properties is None:
            return None
        try:
            return next(
                property for property in self.memory_schema.suggested_properties if property.name == property_id
            )
        except StopIteration:
            return None

    @computed_field
    @property
    def properties(self) -> List[SuggestedProperty]:
        """Get the suggested properties for this section.

        Returns:
            List[SuggestedProperty]: List of suggested properties, or empty list if none.
        """
        return self.memory_schema.suggested_properties or []


class MemoryConfig(BaseModel):
    """Main configuration class for memory.

    Attributes:
        types: List of memory types.
        section_categories: List of section categories.
        sections: List of memory sections.
    """

    types: List[MemoryType] = Field(description="List of memory types")
    section_categories: List[SectionCategory] = Field(description="List of section categories")
    sections: List[Section] = Field(description="List of memory sections")

    def get_section(self, section_id: str) -> Optional[Section]:
        """
        Returns the section with the given ID.

        Args:
            section_id: The ID of the section to get.

        Returns:
            Section: The section with the given ID.
        """
        try:
            return next(section for section in self.sections if section.id == section_id)
        except StopIteration:
            return None

    @computed_field
    @property
    def user_sections(self) -> List[Section]:
        """Get sections that are written by the user.

        Returns:
            List[Section]: List of sections where the category has written_by == "user".
        """
        # Create a mapping of category id to written_by
        category_writers = {cat.id: cat.written_by for cat in self.section_categories}

        # Filter sections where category has written_by == "user"
        return [section for section in self.sections if category_writers.get(section.category) == "user"]

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "MemoryConfig":
        """Creates a MemoryConfig instance from a YAML file.

        Args:
            yaml_path: Path to the YAML configuration file.

        Returns:
            MemoryConfig: An instance of the configuration.

        Raises:
            yaml.YAMLError: If there's an error parsing the YAML file.
            pydantic.ValidationError: If the YAML data doesn't match the expected schema.
            FileNotFoundError: If the YAML file does not exist.
        """
        with open(yaml_path, "r") as f:
            config_data = yaml.safe_load(f)
        return cls(**config_data)

    @classmethod
    def from_yaml_string(cls, yaml_string: str) -> "MemoryConfig":
        """Creates a MemoryConfig instance from a YAML string.

        Args:
            yaml_string: String containing YAML configuration.

        Returns:
            MemoryConfig: An instance of the configuration.

        Raises:
            yaml.YAMLError: If there's an error parsing the YAML string.
            pydantic.ValidationError: If the YAML data doesn't match the expected schema.
        """
        config_data = yaml.safe_load(yaml_string)
        return cls(**config_data)

    @classmethod
    def from_default_config(cls) -> "MemoryConfig":
        """Creates a MemoryConfig instance from the default embedded configuration.

        The default configuration is loaded from the module's resources/configs/memory.yaml file.

        Returns:
            MemoryConfig: An instance of the configuration.

        Raises:
            yaml.YAMLError: If there's an error parsing the YAML file.
            pydantic.ValidationError: If the YAML data doesn't match the expected schema.
            FileNotFoundError: If the default configuration file cannot be found.
        """
        with files("gca_memory_simulator.resources.configs").joinpath("memory.yaml").open("r") as f:
            config_data = yaml.safe_load(f)
        return cls(**config_data)


class PromptParameters(BaseModel):
    """Configuration parameters for a prompt template.

    Attributes:
        temperature: Controls randomness in generation (0.0 to 1.0).
        top_p: Controls diversity via nucleus sampling (0.0 to 1.0).
        seed: Random seed for reproducibility.
        prompt: The actual prompt template text.
        thinking_budget: The thinking budget for generation.
    """

    temperature: float = Field(ge=0.0, le=1.0, description="Temperature for generation")
    top_p: float = Field(ge=0.0, le=1.0, description="Top-p for nucleus sampling")
    seed: int = Field(description="Random seed for reproducibility")
    thinking_budget: int = Field(description="Thinking budget for generation")
    prompt: str = Field(description="The prompt template text")


class MemoryPromptsConfig(BaseModel):
    """Configuration for all prompt templates used in memory processing.

    Attributes:
        parent_system_prompt: Template for the parent system prompt.
        extraction_system_prompt: Template for the extraction system prompt.
        extract: Configuration for extraction prompt.
    """

    parent_system_prompt: str = Field(description="Parent system prompt template")
    extraction_system_prompt: str = Field(description="Extraction system prompt template")
    extract: PromptParameters = Field(description="Extraction prompt config")

    def get_rendered_parent_system_prompt(self, memory_config: MemoryConfig) -> str:
        """Renders the parent system prompt template using the memory configuration.

        Args:
            memory_config: MemoryConfig instance to use for rendering.

        Returns:
            str: The rendered parent system prompt.

        Raises:
            jinja2.TemplateError: If there's an error in template rendering.
        """
        template = Environment(loader=BaseLoader()).from_string(self.parent_system_prompt)
        return template.render(memory=memory_config)

    def get_rendered_extraction_system_prompt(self, memory_config: MemoryConfig) -> str:
        """Renders the extraction system prompt template using the memory configuration.

        Args:
            memory_config: MemoryConfig instance to use for rendering.

        Returns:
            str: The rendered extraction system prompt.

        Raises:
            jinja2.TemplateError: If there's an error in template rendering.
        """
        parent_system_prompt = self.get_rendered_parent_system_prompt(memory_config)
        template = Environment(loader=BaseLoader()).from_string(self.extraction_system_prompt)
        extraction_system_prompt = template.render(memory=memory_config)
        return f"{parent_system_prompt}\n\n## Extraction Instructions\n{extraction_system_prompt}"

    def get_rendered_extract_prompt(self, memory_config: MemoryConfig, user_input: str) -> str:
        """Renders the extract prompt template using the memory configuration and user input.

        Args:
            memory_config: MemoryConfig instance to use for rendering.
            user_input: The user input to extract information from.

        Returns:
            str: The rendered extract prompt.

        Raises:
            jinja2.TemplateError: If there's an error in template rendering.
        """
        template = Environment(loader=BaseLoader()).from_string(self.extract.prompt)
        return template.render(memory=memory_config, user_input=user_input)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "MemoryPromptsConfig":
        """Creates a MemoryPromptsConfig instance from a YAML file.

        Args:
            yaml_path: Path to the YAML configuration file.

        Returns:
            MemoryPromptsConfig: An instance of the configuration.

        Raises:
            yaml.YAMLError: If there's an error parsing the YAML file.
            pydantic.ValidationError: If the YAML data doesn't match the schema.
            FileNotFoundError: If the YAML file does not exist.
        """
        with open(yaml_path, "r") as f:
            config_data = yaml.safe_load(f)
        return cls(**config_data)

    @classmethod
    def from_yaml_string(cls, yaml_string: str) -> "MemoryPromptsConfig":
        """Creates a MemoryPromptsConfig instance from a YAML string.

        Args:
            yaml_string: String containing YAML configuration.

        Returns:
            MemoryPromptsConfig: An instance of the configuration.

        Raises:
            yaml.YAMLError: If there's an error parsing the YAML string.
            pydantic.ValidationError: If the YAML data doesn't match the schema.
        """
        config_data = yaml.safe_load(yaml_string)
        return cls(**config_data)

    @classmethod
    def from_default_config(cls) -> "MemoryPromptsConfig":
        """Creates a MemoryPromptsConfig instance from the default configuration.

        The default configuration is loaded from resources/configs/memory_prompts.yaml.

        Returns:
            MemoryPromptsConfig: An instance of the configuration.

        Raises:
            yaml.YAMLError: If there's an error parsing the YAML file.
            pydantic.ValidationError: If the YAML data doesn't match the schema.
            FileNotFoundError: If the default configuration file cannot be found.
        """
        with files("gca_memory_simulator.resources.configs").joinpath("memory_prompts.yaml").open("r") as f:
            config_data = yaml.safe_load(f)
        return cls(**config_data)


class RenderedMemoryPromptsConfig(BaseModel):
    """Configuration for all rendered prompt templates used in memory processing.

    Attributes:
        prompts_config: The original prompts config.
        rendered_parent_system_prompt: Rendered parent system prompt.
        rendered_extraction_system_prompt: Rendered extraction system prompt.
        extract_config: Configuration for extraction.
    """

    prompts_config: MemoryPromptsConfig = Field(description="Original prompts config")
    rendered_parent_system_prompt: str = Field(description="Rendered parent system prompt")
    rendered_extraction_system_prompt: str = Field(description="Rendered extraction system prompt")
    extract_config: PromptParameters = Field(description="Extract prompt config")

    def get_rendered_extract_prompt(self, memory_config: MemoryConfig, user_input: str) -> str:
        """Renders the extract prompt template using the memory configuration and user input.

        Args:
            memory_config: MemoryConfig instance to use for rendering.
            user_input: The user input to extract information from.

        Returns:
            str: The rendered extract prompt.

        Raises:
            jinja2.TemplateError: If there's an error in template rendering.
        """
        return self.prompts_config.get_rendered_extract_prompt(memory_config, user_input)

    @classmethod
    def from_prompts_config(
        cls, prompts_config: MemoryPromptsConfig, memory_config: MemoryConfig
    ) -> "RenderedMemoryPromptsConfig":
        """Creates a RenderedMemoryPromptsConfig instance from a MemoryPromptsConfig instance.

        Args:
            prompts_config: MemoryPromptsConfig instance to use for rendering.
            memory_config: MemoryConfig instance to use for rendering.

        Returns:
            RenderedMemoryPromptsConfig: An instance with rendered prompts.
        """
        return cls(
            prompts_config=prompts_config,
            rendered_parent_system_prompt=prompts_config.get_rendered_parent_system_prompt(memory_config),
            rendered_extraction_system_prompt=prompts_config.get_rendered_extraction_system_prompt(memory_config),
            extract_config=prompts_config.extract,
        )

    @classmethod
    def from_yaml(
        cls, yaml_path: str, memory_config: MemoryConfig
    ) -> tuple["MemoryPromptsConfig", "RenderedMemoryPromptsConfig"]:
        """Creates both MemoryPromptsConfig and RenderedMemoryPromptsConfig instances from a YAML file.

        Args:
            yaml_path: Path to the YAML configuration file.
            memory_config: MemoryConfig instance to use for rendering.

        Returns:
            tuple: A tuple containing (MemoryPromptsConfig, RenderedMemoryPromptsConfig).

        Raises:
            yaml.YAMLError: If there's an error parsing the YAML file.
            pydantic.ValidationError: If the YAML data doesn't match the schema.
            FileNotFoundError: If the YAML file does not exist.
        """
        prompts_config = MemoryPromptsConfig.from_yaml(yaml_path)
        rendered_config = cls.from_prompts_config(prompts_config, memory_config)
        return prompts_config, rendered_config

    @classmethod
    def from_default_config(
        cls, memory_config: MemoryConfig
    ) -> tuple["MemoryPromptsConfig", "RenderedMemoryPromptsConfig"]:
        """Creates both MemoryPromptsConfig and RenderedMemoryPromptsConfig instances from the default configuration.

        Args:
            memory_config: MemoryConfig instance to use for rendering.

        Returns:
            tuple: A tuple containing (MemoryPromptsConfig, RenderedMemoryPromptsConfig).

        Raises:
            yaml.YAMLError: If there's an error parsing the YAML file.
            pydantic.ValidationError: If the YAML data doesn't match the schema.
            FileNotFoundError: If the default configuration file cannot be found.
        """
        prompts_config = MemoryPromptsConfig.from_default_config()
        rendered_config = cls.from_prompts_config(prompts_config, memory_config)
        return prompts_config, rendered_config


def create_dynamic_user_memory_model(memory_config: MemoryConfig) -> Type[BaseModel]:
    """Creates a dynamic Pydantic model that represents all user-written sections in memory.yaml.

    This function dynamically generates a Pydantic model at runtime where:
    - Each user section becomes a property in the model
    - The property name is the section ID
    - Each property has its own dynamically generated type based on the section's suggested_properties
    - Sections with corpus instead of suggested_properties are skipped with a logged error

    Args:
        memory_config: MemoryConfig instance containing the memory configuration.

    Returns:
        Type[BaseModel]: A dynamically created Pydantic model class.

    Raises:
        ValueError: If no user sections are found or if all user sections are skipped.
    """

    # Get all user sections
    user_sections = memory_config.user_sections

    if not user_sections:
        raise ValueError("No user sections found in memory configuration")

    # Dictionary to hold the fields for the main model
    main_model_fields: Dict[str, Any] = {}

    for section in user_sections:
        # Check if section has corpus instead of suggested_properties
        if section.memory_schema.corpus is not None:
            logger.error(
                f"Skipping section '{section.id}' because it has corpus configuration "
                f"instead of suggested_properties"
            )
            continue

        # Check if section has suggested_properties
        if not section.memory_schema.suggested_properties:
            logger.error(f"Skipping section '{section.id}' because it has no suggested_properties")
            continue

        # Create dynamic model for this section
        section_model_name = section.name.replace(" ", "").replace("-", "").replace("_", "")
        section_fields: Dict[str, Any] = {}

        for prop in section.memory_schema.suggested_properties:
            # Determine the Python type for this property
            if prop.type == "list":
                prop_type = List[str]
                default_value = Field(default_factory=list, description=prop.description)
            else:
                prop_type = Optional[str]
                default_value = Field(default=None, description=prop.description)

            section_fields[prop.name] = (prop_type, default_value)

        # Create the section model
        section_model = create_model(
            section_model_name,
            **section_fields,
            __base__=BaseModel,
            __module__=__name__,
            __doc__=f"User memory section: {section.name}",
        )

        # Add this section model as a field in the main model
        main_model_fields[section.id] = (
            Optional[section_model],
            Field(default=None, description=f"User memory section: {section.name}"),
        )

    if not main_model_fields:
        raise ValueError("No valid user sections found - all sections were skipped")

    # Create the main model
    main_model = create_model(
        "UserMemoryModel",
        **main_model_fields,
        __base__=BaseModel,
        __module__=__name__,
        __doc__="Layout of the Agent Memory parts that are written by the interactions with the user",
    )

    logger.info(f"Created dynamic user memory model with {len(main_model_fields)} sections")

    return main_model
